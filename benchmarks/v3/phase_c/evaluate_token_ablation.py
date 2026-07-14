# 作用：在固定 v3 prompt、尺度和 LOIO 面积阈值下比较不同 superpixel token 的 retrieval 表现，不训练模型。

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import binary_segmentation_metrics, predict_top_area, safe_l2_normalize
from benchmarks.v3.phase_c.recompute_v2style_bestarea_metrics import loio_bestarea_priors
from benchmarks.v3.phase_c.run_multiscale_baseline import (
    METRICS_PATH, MULTISCALE_ROOT, NUM_CLASSES, PROMPT_PATH, SCALES, ScaleData,
    load_scale_data, max_cosine_scores, prompt_segments_for_scale, ranking_metrics, read_csv,
)

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
STAGE_ROOT = V3_ROOT / "pret_superpixel" / "_stage_a_generation"
OUT_METRICS = V3_ROOT / "pret_superpixel" / "evaluations" / "token_ablation_metrics.csv"
OUT_SUMMARY = V3_ROOT / "pret_superpixel" / "evaluations" / "token_ablation_summary.csv"
REPORT_PATH = Path(__file__).resolve().parent / "token_ablation_report.md"

VARIANTS = (
    "image_only",
    "cell_reg",
    "image_cell_reg_cellw0p5",
    "image_cell_reg_texture_cellstats",
    "image_patch_cell_reg",
)
FIELDS = ["variant", "query_id", "image_id", "target_class", "scale", "prompt_quality", "prompt_mode", "mAP", "AUROC", "BestDice", "Dice_classwise_toparea", "mIoU", "Precision", "Recall", "score_std", "area_fraction", "status"]
SUMMARY_FIELDS = ["variant", "scale", "prompt_quality", "n", "mAP", "AUROC", "BestDice", "Dice_classwise_toparea", "mIoU", "Precision", "Recall", "score_std"]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    os.replace(tmp, path)


def load_variant(image_id: str, scale: str, variant: str) -> ScaleData:
    data = load_scale_data(image_id, scale)
    token_path = STAGE_ROOT / scale / "pret_superpixel" / image_id / f"tokens_{variant}.npy"
    tokens = np.asarray(np.load(token_path, mmap_mode="r"), dtype=np.float32)
    if tokens.shape[0] != len(data.segment_ids):
        raise ValueError(f"{image_id}/{scale}/{variant}: token rows do not match superpixels")
    data.tokens = tokens
    data.tokens_norm = safe_l2_normalize(tokens, axis=1)
    return data


def evaluate(prompt: dict[str, str], data: ScaleData, variant: str, area_fraction: float, lambda_neg: float) -> dict[str, object]:
    target_class = int(prompt["target_class"])
    pos_ids, neg_ids = prompt_segments_for_scale(prompt, data)
    hard = data.gt_majority_label == target_class
    candidate = data.valid
    base = {"variant": variant, "query_id": prompt["query_id"], "image_id": prompt["image_id"], "target_class": target_class, "scale": data.scale, "prompt_quality": prompt["prompt_quality"], "prompt_mode": prompt["prompt_mode"], "area_fraction": area_fraction}
    if not pos_ids or not np.any(candidate & hard) or not np.any(candidate & ~hard):
        return {**base, "status": "invalid_query"}
    pos = max_cosine_scores(data.tokens_norm, pos_ids)
    neg = max_cosine_scores(data.tokens_norm, neg_ids)
    score = pos if not neg_ids else pos - lambda_neg * neg
    scores, labels, areas = score[candidate], hard[candidate], data.areas[candidate]
    ap, auc = ranking_metrics(labels, scores)
    pred = predict_top_area(scores, areas, area_fraction)
    metric = binary_segmentation_metrics(labels, pred)
    best = 0.0
    for fraction in np.linspace(0.01, 0.60, 60):
        best = max(best, float(binary_segmentation_metrics(labels, predict_top_area(scores, areas, float(fraction)))["dice"]))
    return {**base, "mAP": ap, "AUROC": auc, "BestDice": best, "Dice_classwise_toparea": float(metric["dice"]), "mIoU": float(metric["iou"]), "Precision": float(metric["precision"]), "Recall": float(metric["recall"]), "score_std": float(np.std(scores)), "status": "ok"}


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(str(row["variant"]), str(row["scale"]), str(row["prompt_quality"]))].append(row)
    output = []
    for (variant, scale, quality), group in sorted(groups.items()):
        mean = lambda field: float(np.mean([float(row[field]) for row in group]))
        output.append({"variant": variant, "scale": scale, "prompt_quality": quality, "n": len(group), **{field: mean(field) for field in SUMMARY_FIELDS[4:]}})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--max_queries", type=int, default=0)
    args = parser.parse_args()
    prompts = read_csv(PROMPT_PATH)
    if args.max_queries:
        prompts = prompts[:args.max_queries]
    baseline_rows = read_csv(METRICS_PATH)
    priors = loio_bestarea_priors(baseline_rows)
    # Degenerate target rows do not have an image-specific LOIO entry.  Keep the
    # ablation complete by falling back to the available class-scale median.
    fallback_priors: dict[tuple[int, str], float] = {}
    by_class_scale: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in baseline_rows:
        if row.get("status") == "ok":
            by_class_scale[(int(row["target_class"]), row["scale"])].append(float(row["BestArea"]))
    for key, values in by_class_scale.items():
        fallback_priors[key] = float(np.clip(np.median(values), 0.01, 0.60))
    cache: dict[tuple[str, str, str], ScaleData] = {}
    rows: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts, start=1):
        for variant in args.variants:
            for scale in SCALES:
                key = (prompt["image_id"], scale, variant)
                if key not in cache:
                    cache[key] = load_variant(*key)
                prior = priors.get(
                    (prompt["image_id"], int(prompt["target_class"]), scale),
                    fallback_priors.get((int(prompt["target_class"]), scale), 0.18),
                )
                rows.append(evaluate(prompt, cache[key], variant, prior, args.lambda_neg))
        if index % 100 == 0 or index == len(prompts):
            print(f"token_ablation {index}/{len(prompts)}", flush=True)
    summary = summarize(rows)
    write_csv(OUT_METRICS, rows, FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    lines = ["# Phase C Token Ablation", "", f"- queries: {len(prompts)}", "- variants: " + ", ".join(args.variants), "", "| token | scale | prompt | n | mAP | AUROC | BestDice | Dice | mIoU |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['scale']} | {row['prompt_quality']} | {row['n']} | {row['mAP']:.4f} | {row['AUROC']:.4f} | {row['BestDice']:.4f} | {row['Dice_classwise_toparea']:.4f} | {row['mIoU']:.4f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
