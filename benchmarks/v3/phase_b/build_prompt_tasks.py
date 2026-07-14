# 作用：基于 v3 Phase A 的 medium superpixel 重新生成 prompt-task 数据与验证报告。

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST_PATH = PRET_ROOT / "data_manifest_v3.csv"
MULTISCALE_ROOT = PRET_ROOT / "multiscale_tokens"
PROMPT_TASK_DIR = PRET_ROOT / "prompt_tasks"
REPORTS_DIR = PRET_ROOT / "reports"
VIS_DIR = PRET_ROOT / "visualizations" / "phase_b_prompt_task_samples"
PHASE_DIR = Path(__file__).resolve().parent
LEGACY_PROMPT_CSV = Path("/nfs-medical3/zyh/pret_eval_prompt_class_fix_20260706_gt_pure_prompts/prompts.csv")

AUTO_PROMPT_PATH = PROMPT_TASK_DIR / "auto_prompt_tasks.csv"
ALL_PROMPT_PATH = PROMPT_TASK_DIR / "all_prompt_tasks.csv"
SUMMARY_PATH = PROMPT_TASK_DIR / "prompt_task_summary.csv"
LEGACY_AUDIT_PATH = PROMPT_TASK_DIR / "legacy_prompt_audit.csv"
VALIDATION_PATH = REPORTS_DIR / "phase_b_validation.json"
REPORT_PATH = PHASE_DIR / "report.md"

TASK_FIELDS = [
    "query_id",
    "image_id",
    "target_class",
    "scale",
    "prompt_source",
    "prompt_mode",
    "prompt_quality",
    "shot",
    "positive_segment_count",
    "negative_segment_count",
    "positive_segments",
    "negative_segments",
    "negative_gt_majority_labels",
    "x0_original",
    "y0_original",
    "x1_original",
    "y1_original",
    "positive_boxes",
    "negative_boxes",
    "prompt_purity",
    "prompt_area_10x_pixels",
    "prompt_target_area_fraction",
    "prompt_valid_area_fraction",
    "hard_negative_similarity_mean",
    "hard_negative_similarity_max",
]

SUMMARY_FIELDS = [
    "target_class",
    "clean_positive_queries",
    "noisy_positive_queries",
    "hard_negative_queries",
    "positive_only_queries",
    "positive_negative_queries",
    "total_queries",
]


@dataclass(frozen=True)
class Segment:
    image_id: str
    segment_id: int
    area: int
    center_x: float
    center_y: float
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    cell_count: int
    cell_density: float
    gt_majority_label: int
    gt_purity: float
    valid_fraction: float

    @property
    def box(self) -> str:
        return f"{self.bbox_x0}:{self.bbox_y0}:{self.bbox_x1}:{self.bbox_y1}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def stable_rank(items: Iterable[Segment], *parts: object) -> list[Segment]:
    prefix = "|".join(str(part) for part in parts)
    return sorted(
        items,
        key=lambda item: hashlib.sha256(f"{prefix}|{item.segment_id}".encode("utf-8")).hexdigest(),
    )


def parse_segment(row: dict[str, str], image_id: str) -> Segment:
    return Segment(
        image_id=image_id,
        segment_id=int(float(row["segment_id"])),
        area=int(float(row.get("area", 0))),
        center_x=float(row.get("center_x", 0.0)),
        center_y=float(row.get("center_y", 0.0)),
        bbox_x0=int(float(row["bbox_x0"])),
        bbox_y0=int(float(row["bbox_y0"])),
        bbox_x1=int(float(row["bbox_x1"])),
        bbox_y1=int(float(row["bbox_y1"])),
        cell_count=int(float(row.get("cell_count", 0))),
        cell_density=float(row.get("cell_density", 0.0)),
        gt_majority_label=int(float(row.get("gt_majority_label", 255))),
        gt_purity=float(row.get("gt_purity", row.get("gt_target_fraction", 0.0))),
        valid_fraction=float(row.get("valid_fraction", 0.0)),
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def weighted_mean(rows: list[Segment], attr: str) -> float:
    total = sum(max(row.area, 1) for row in rows)
    if total <= 0:
        return 0.0
    return float(sum(getattr(row, attr) * max(row.area, 1) for row in rows) / total)


def make_task(
    image_id: str,
    target_class: int,
    quality: str,
    prompt_mode: str,
    positive: list[Segment],
    negative: list[Segment],
    similarity: dict[int, float],
    index: int,
) -> dict[str, object]:
    if not positive:
        raise ValueError("each query must have at least one positive prompt")
    positive_segments = [row.segment_id for row in positive]
    negative_segments = [row.segment_id for row in negative]
    negative_labels = [row.gt_majority_label for row in negative]
    sim_values = [similarity.get(row.segment_id, 0.0) for row in negative]
    query_id = f"{image_id}_c{target_class}_msp_{quality}_{index:03d}"
    return {
        "query_id": query_id,
        "image_id": image_id,
        "target_class": target_class,
        "scale": "medium",
        "prompt_source": "phase_b_medium_superpixel",
        "prompt_mode": prompt_mode,
        "prompt_quality": quality,
        "shot": len(positive),
        "positive_segment_count": len(positive),
        "negative_segment_count": len(negative),
        "positive_segments": ";".join(str(value) for value in positive_segments),
        "negative_segments": ";".join(str(value) for value in negative_segments),
        "negative_gt_majority_labels": ";".join(str(value) for value in negative_labels),
        "x0_original": positive[0].bbox_x0,
        "y0_original": positive[0].bbox_y0,
        "x1_original": positive[0].bbox_x1,
        "y1_original": positive[0].bbox_y1,
        "positive_boxes": ";".join(row.box for row in positive),
        "negative_boxes": ";".join(row.box for row in negative),
        "prompt_purity": weighted_mean(positive, "gt_purity"),
        "prompt_area_10x_pixels": sum(row.area for row in positive),
        "prompt_target_area_fraction": weighted_mean(positive, "gt_purity"),
        "prompt_valid_area_fraction": weighted_mean(positive, "valid_fraction"),
        "hard_negative_similarity_mean": float(np.mean(sim_values)) if sim_values else "",
        "hard_negative_similarity_max": float(np.max(sim_values)) if sim_values else "",
    }


def select_hard_negatives(
    target_class: int,
    clean: list[Segment],
    positive_pool: list[Segment],
    negative_pool: list[Segment],
    tokens: np.ndarray,
    limit: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[list[Segment], dict[int, float]]:
    if not negative_pool:
        return [], {}
    prototype_pool = clean or positive_pool
    if not prototype_pool:
        return [], {}
    token_norm = normalize_rows(tokens)
    prototype_ids = [row.segment_id for row in prototype_pool if row.segment_id < token_norm.shape[0]]
    if not prototype_ids:
        return [], {}
    prototype = token_norm[prototype_ids].mean(axis=0, keepdims=True)
    prototype = normalize_rows(prototype)[0]
    scores: dict[int, float] = {}
    for row in negative_pool:
        if row.segment_id >= token_norm.shape[0]:
            continue
        scores[row.segment_id] = float(np.dot(token_norm[row.segment_id], prototype))
    candidates = [row for row in negative_pool if row.segment_id in scores and row.gt_majority_label != target_class]
    if not candidates:
        return [], scores
    candidate_scores = np.asarray([scores[row.segment_id] for row in candidates], dtype=np.float32)
    low, high = np.quantile(candidate_scores, [low_quantile, high_quantile])
    bounded = [row for row in candidates if low <= scores[row.segment_id] <= high]
    # 采样中高难度异类，排除最极端、几乎与正提示重复的负例。
    selected = stable_rank(bounded or candidates, target_class, "bounded_hard_negative")
    return selected[:limit], scores


def build_tasks_for_image(image_id: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    medium_dir = MULTISCALE_ROOT / image_id / "medium"
    csv_path = medium_dir / "superpixels.csv"
    token_path = medium_dir / "tokens_image_cell_reg.npy"
    segments_path = medium_dir / "superpixels.npy"
    missing = [str(path) for path in (csv_path, token_path, segments_path) if not path.exists()]
    if missing:
        return [], {"image_id": image_id, "status": "missing_inputs", "missing": missing}

    segments = [parse_segment(row, image_id) for row in read_csv(csv_path)]
    tokens = np.load(token_path, mmap_mode="r")
    by_label: dict[int, list[Segment]] = defaultdict(list)
    for segment in segments:
        if 0 <= segment.gt_majority_label < 255 and segment.valid_fraction > 0:
            by_label[segment.gt_majority_label].append(segment)

    tasks: list[dict[str, object]] = []
    class_reports: list[dict[str, object]] = []
    for target_class in sorted(by_label):
        clean_pool = [
            row
            for row in by_label[target_class]
            if row.gt_purity >= args.clean_min_purity and row.valid_fraction >= args.clean_min_valid_fraction
        ]
        noisy_pool = [
            row
            for row in by_label[target_class]
            if args.noisy_min_purity <= row.gt_purity < args.noisy_max_purity
            and row.valid_fraction >= args.noisy_min_valid_fraction
        ]
        positive_pool = [
            row
            for row in by_label[target_class]
            if row.gt_purity >= args.noisy_min_purity and row.valid_fraction >= args.noisy_min_valid_fraction
        ]
        negative_pool = [
            row
            for row in segments
            if row.gt_majority_label != target_class
            and row.gt_majority_label < 255
            and row.valid_fraction >= args.negative_min_valid_fraction
        ]

        clean = sorted(stable_rank(clean_pool, image_id, target_class, "clean"), key=lambda row: (-row.gt_purity, row.segment_id))
        noisy = sorted(stable_rank(noisy_pool, image_id, target_class, "noisy"), key=lambda row: (-row.gt_purity, row.segment_id))
        clean = clean[: args.max_clean]
        noisy = noisy[: args.max_noisy]
        hard_negatives, similarity = select_hard_negatives(
            target_class,
            clean,
            positive_pool,
            negative_pool,
            np.asarray(tokens),
            args.max_hard_negative,
            args.hard_negative_low_quantile,
            args.hard_negative_high_quantile,
        )

        clean_count = 0
        noisy_count = 0
        hard_count = 0
        for segment in clean:
            clean_count += 1
            tasks.append(make_task(image_id, target_class, "clean", "positive_only", [segment], [], similarity, clean_count))
        for segment in noisy:
            noisy_count += 1
            tasks.append(make_task(image_id, target_class, "noisy", "positive_only", [segment], [], similarity, noisy_count))
        if clean and hard_negatives:
            for index, negative in enumerate(hard_negatives, start=1):
                positive = clean[(index - 1) % len(clean)]
                neg_count = min(args.max_negative_prompts_per_query, len(hard_negatives), 1 + ((index - 1) % args.max_negative_prompts_per_query))
                neg_start = index - 1
                negatives = [hard_negatives[(neg_start + offset) % len(hard_negatives)] for offset in range(neg_count)]
                hard_count += 1
                tasks.append(
                    make_task(
                        image_id,
                        target_class,
                        "hard_negative",
                        "positive_negative",
                        [positive],
                        negatives,
                        similarity,
                        hard_count,
                    )
                )

        class_reports.append(
            {
                "target_class": target_class,
                "clean_candidate_count": len(clean_pool),
                "noisy_candidate_count": len(noisy_pool),
                "hard_negative_candidate_count": len(negative_pool),
                "clean_selected": clean_count,
                "noisy_selected": noisy_count,
                "hard_negative_selected": hard_count,
            }
        )
    return tasks, {"image_id": image_id, "status": "ok", "classes": class_reports}


def summarize_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        by_class[int(task["target_class"])].append(task)
    rows: list[dict[str, object]] = []
    for target_class in sorted(by_class):
        group = by_class[target_class]
        quality_counts = Counter(row["prompt_quality"] for row in group)
        mode_counts = Counter(row["prompt_mode"] for row in group)
        rows.append(
            {
                "target_class": target_class,
                "clean_positive_queries": quality_counts["clean"],
                "noisy_positive_queries": quality_counts["noisy"],
                "hard_negative_queries": quality_counts["hard_negative"],
                "positive_only_queries": mode_counts["positive_only"],
                "positive_negative_queries": mode_counts["positive_negative"],
                "total_queries": len(group),
            }
        )
    return rows


def write_legacy_audit() -> dict[str, object]:
    if not LEGACY_PROMPT_CSV.exists():
        write_csv(LEGACY_AUDIT_PATH, [], ["usage_status"])
        return {"legacy_prompt_path": str(LEGACY_PROMPT_CSV), "exists": False, "rows": 0}
    rows = read_csv(LEGACY_PROMPT_CSV)
    fields = ["usage_status", "legacy_prompt_path"] + list(rows[0].keys() if rows else [])
    audit_rows = [{"usage_status": "not_used", "legacy_prompt_path": str(LEGACY_PROMPT_CSV), **row} for row in rows]
    write_csv(LEGACY_AUDIT_PATH, audit_rows, fields)
    return {"legacy_prompt_path": str(LEGACY_PROMPT_CSV), "exists": True, "rows": len(rows), "usage_status": "not_used"}


def parse_id_list(value: object) -> list[int]:
    text = str(value or "")
    if not text:
        return []
    return [int(part) for part in text.split(";") if part != ""]


def validate_tasks(tasks: list[dict[str, object]], summary_rows: list[dict[str, object]], image_reports: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    query_ids = [str(row["query_id"]) for row in tasks]
    if len(query_ids) != len(set(query_ids)):
        errors.append("query_id is not unique")
    if not tasks:
        errors.append("no prompt tasks generated")
    for path in (AUTO_PROMPT_PATH, ALL_PROMPT_PATH, SUMMARY_PATH, LEGACY_AUDIT_PATH):
        if not path.exists():
            errors.append(f"missing output file: {path}")
    if (PROMPT_TASK_DIR / "expert_prompt_tasks.csv").exists():
        errors.append("expert_prompt_tasks.csv exists but Phase B should not create formal expert prompts")

    for row in tasks:
        positives = parse_id_list(row["positive_segments"])
        negatives = parse_id_list(row["negative_segments"])
        labels = parse_id_list(row["negative_gt_majority_labels"])
        target = int(row["target_class"])
        purity = float(row["prompt_purity"])
        valid = float(row["prompt_valid_area_fraction"])
        if not positives or int(row["positive_segment_count"]) < 1:
            errors.append(f"{row['query_id']} has no positive prompt")
        if len(negatives) != int(row["negative_segment_count"]):
            errors.append(f"{row['query_id']} negative count mismatch")
        if labels and any(label == target for label in labels):
            errors.append(f"{row['query_id']} has target-class hard negative")
        if row["prompt_quality"] == "clean" and (purity < 0.8 or valid < 0.8):
            errors.append(f"{row['query_id']} violates clean thresholds")
        if row["prompt_quality"] == "noisy" and not (0.5 <= purity < 0.8 and valid >= 0.5):
            errors.append(f"{row['query_id']} violates noisy thresholds")
        if row["prompt_quality"] == "hard_negative":
            if not negatives:
                errors.append(f"{row['query_id']} has no hard negative")
            if any(label == target for label in labels):
                errors.append(f"{row['query_id']} hard negative label equals target")

    class_coverage = {
        int(row["target_class"]): {
            "clean": int(row["clean_positive_queries"]),
            "noisy": int(row["noisy_positive_queries"]),
            "hard_negative": int(row["hard_negative_queries"]),
            "total": int(row["total_queries"]),
        }
        for row in summary_rows
    }
    return {
        "passed": not errors,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "auto_prompt_tasks": str(AUTO_PROMPT_PATH),
        "all_prompt_tasks": str(ALL_PROMPT_PATH),
        "prompt_task_summary": str(SUMMARY_PATH),
        "legacy_prompt_audit": str(LEGACY_AUDIT_PATH),
        "formal_expert_prompt_tasks_created": False,
        "total_queries": len(tasks),
        "unique_query_ids": len(set(query_ids)),
        "images": len({row["image_id"] for row in tasks}),
        "classes": len(class_coverage),
        "class_coverage": class_coverage,
        "image_reports": image_reports,
        "errors": errors[:50],
    }


def sample_visual_tasks(tasks: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        by_class[int(task["target_class"])].append(task)
    selected: list[dict[str, object]] = []
    for target_class in sorted(by_class):
        ranked = sorted(by_class[target_class], key=lambda row: (row["prompt_quality"], row["query_id"]))
        selected.extend(ranked[: max(1, limit // max(1, len(by_class)))])
    if len(selected) < limit:
        seen = {row["query_id"] for row in selected}
        for task in sorted(tasks, key=lambda row: row["query_id"]):
            if task["query_id"] not in seen:
                selected.append(task)
                seen.add(task["query_id"])
            if len(selected) >= limit:
                break
    return selected[:limit]


def parse_boxes(text: object) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for part in str(text or "").split(";"):
        if not part:
            continue
        x0, y0, x1, y1 = [int(float(value)) for value in part.split(":")]
        boxes.append((x0, y0, x1, y1))
    return boxes


def write_visualizations(tasks: list[dict[str, object]], limit: int) -> dict[str, object]:
    if limit <= 0:
        return {"enabled": False, "rows": 0}
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {"enabled": False, "rows": 0, "error": "PIL is not installed"}

    if VIS_DIR.exists():
        shutil.rmtree(VIS_DIR)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for task in sample_visual_tasks(tasks, limit):
        image_id = str(task["image_id"])
        image_path = MULTISCALE_ROOT / image_id / "medium" / "he_10x_rgb.npy"
        if not image_path.exists():
            continue
        image = np.load(image_path, mmap_mode="r")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            continue
        max_dim = max(image.shape[:2])
        scale = min(1.0, 1800.0 / max_dim)
        if scale < 1.0:
            pil = Image.fromarray(image[:, :, :3].astype(np.uint8)).resize(
                (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                resample=Image.Resampling.BILINEAR,
            )
        else:
            pil = Image.fromarray(image[:, :, :3].astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        for box in parse_boxes(task["positive_boxes"]):
            scaled = tuple(int(round(value * scale)) for value in box)
            draw.rectangle(scaled, outline=(0, 255, 0), width=4)
        for box in parse_boxes(task["negative_boxes"]):
            scaled = tuple(int(round(value * scale)) for value in box)
            draw.rectangle(scaled, outline=(255, 0, 0), width=4)
        out_name = f"{task['query_id']}.png".replace("/", "_")
        out_path = VIS_DIR / out_name
        pil.save(out_path)
        index_rows.append(
            {
                "query_id": task["query_id"],
                "image_id": image_id,
                "target_class": task["target_class"],
                "prompt_quality": task["prompt_quality"],
                "path": str(out_path),
            }
        )
    write_csv(VIS_DIR / "index.csv", index_rows, ["query_id", "image_id", "target_class", "prompt_quality", "path"])
    return {"enabled": True, "rows": len(index_rows), "path": str(VIS_DIR)}


def write_report(validation: dict[str, object], legacy_audit: dict[str, object], visual_report: dict[str, object]) -> None:
    coverage = validation.get("class_coverage", {})
    lines = [
        "# Phase B Prompt-task Regeneration",
        "",
        "## Summary",
        "",
        "Phase B was regenerated from the current `lowmag_loose + slic` multiscale superpixels.",
        "The formal prompt task CSVs do not include the old prompt CSV; it is retained only as `legacy_prompt_audit.csv` with `usage_status=not_used`.",
        "",
        "## Outputs",
        "",
        f"- `auto_prompt_tasks.csv`: {validation['auto_prompt_tasks']}",
        f"- `all_prompt_tasks.csv`: {validation['all_prompt_tasks']}",
        f"- `prompt_task_summary.csv`: {validation['prompt_task_summary']}",
        f"- `phase_b_validation.json`: {VALIDATION_PATH}",
        f"- sample visualizations: {visual_report.get('path', '')}",
        "",
        "## Validation",
        "",
        f"- passed: {validation['passed']}",
        f"- total_queries: {validation['total_queries']}",
        f"- unique_query_ids: {validation['unique_query_ids']}",
        f"- images: {validation['images']}",
        f"- classes: {validation['classes']}",
        f"- legacy prompt rows audited, not used: {legacy_audit.get('rows', 0)}",
        "",
        "## Class Coverage",
        "",
        "| class | clean | noisy | hard-negative | total |",
        "|---:|---:|---:|---:|---:|",
    ]
    for class_id in sorted(int(key) for key in coverage.keys()):
        row = coverage[class_id]
        lines.append(f"| {class_id} | {row['clean']} | {row['noisy']} | {row['hard_negative']} | {row['total']} |")
    if validation.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in validation["errors"])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--max_clean", type=int, default=10)
    parser.add_argument("--max_noisy", type=int, default=5)
    parser.add_argument("--max_hard_negative", type=int, default=10)
    parser.add_argument("--max_negative_prompts_per_query", type=int, default=3)
    parser.add_argument("--clean_min_purity", type=float, default=0.8)
    parser.add_argument("--clean_min_valid_fraction", type=float, default=0.8)
    parser.add_argument("--noisy_min_purity", type=float, default=0.5)
    parser.add_argument("--noisy_max_purity", type=float, default=0.8)
    parser.add_argument("--noisy_min_valid_fraction", type=float, default=0.5)
    parser.add_argument("--negative_min_valid_fraction", type=float, default=0.5)
    parser.add_argument("--hard_negative_low_quantile", type=float, default=0.85)
    parser.add_argument("--hard_negative_high_quantile", type=float, default=0.95)
    parser.add_argument("--max_visualizations", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.hard_negative_low_quantile < args.hard_negative_high_quantile <= 1:
        raise ValueError("hard negative quantiles must satisfy 0 <= low < high <= 1")
    manifest_rows = read_csv(args.manifest)
    all_tasks: list[dict[str, object]] = []
    image_reports: list[dict[str, object]] = []
    for row in manifest_rows:
        image_id = row["image_id"]
        tasks, report = build_tasks_for_image(image_id, args)
        all_tasks.extend(tasks)
        image_reports.append(report)
        print(f"phase_b image_id={image_id} tasks={len(tasks)} status={report['status']}", flush=True)

    all_tasks.sort(key=lambda row: (str(row["image_id"]), int(row["target_class"]), str(row["prompt_quality"]), str(row["query_id"])))
    summary_rows = summarize_tasks(all_tasks)
    write_csv(AUTO_PROMPT_PATH, all_tasks, TASK_FIELDS)
    write_csv(ALL_PROMPT_PATH, all_tasks, TASK_FIELDS)
    write_csv(SUMMARY_PATH, summary_rows, SUMMARY_FIELDS)
    legacy_audit = write_legacy_audit()
    validation = validate_tasks(all_tasks, summary_rows, image_reports)
    write_json(VALIDATION_PATH, validation)
    visual_report = write_visualizations(all_tasks, args.max_visualizations)
    write_report(validation, legacy_audit, visual_report)
    if not validation["passed"]:
        raise SystemExit(f"Phase B validation failed: {validation['errors'][:5]}")
    print(json.dumps({"passed": True, "total_queries": len(all_tasks), "validation": str(VALIDATION_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
