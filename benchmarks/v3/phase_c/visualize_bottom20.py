# 作用：可视化 Phase C baseline 中 Dice_classwise_toparea 最低的后 20% query-scale 结果。

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import predict_top_area


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MULTISCALE_ROOT = PRET_ROOT / "multiscale_tokens"
PROMPT_PATH = PRET_ROOT / "prompt_tasks" / "all_prompt_tasks.csv"
METRICS_PATH = PRET_ROOT / "evaluations" / "multiscale_baseline_metrics.csv"
SCORE_DIR = PRET_ROOT / "evaluations" / "query_scale_scores"
VIS_ROOT = PRET_ROOT / "visualizations" / "phase_c_bottom20"
REPORT_PATH = Path(__file__).resolve().parent / "bottom20_visual_report.md"

FIELDS = [
    "rank",
    "query_id",
    "image_id",
    "target_class",
    "scale",
    "prompt_quality",
    "prompt_mode",
    "Dice_classwise_toparea",
    "BestDice",
    "mAP",
    "AUROC",
    "PredArea",
    "GTArea",
    "Precision",
    "Recall",
    "status",
    "visual_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
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


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    boxes = []
    for item in str(text or "").split(";"):
        if not item:
            continue
        boxes.append(tuple(int(float(value)) for value in item.split(":")))
    return boxes


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def target_bbox(segments: np.ndarray, mask_by_segment: np.ndarray, max_segments: int = 80) -> tuple[int, int, int, int] | None:
    ids = np.flatnonzero(mask_by_segment)
    if ids.size == 0:
        return None
    if ids.size > max_segments:
        ids = ids[:max_segments]
    full_mask = np.isin(segments, ids)
    yy, xx = np.nonzero(full_mask)
    if yy.size == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def union_crop(
    boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    margin: int,
    max_side: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    valid = [box for box in boxes if box is not None]
    if not valid:
        return 0, 0, min(width, max_side), min(height, max_side)
    x0 = max(0, min(box[0] for box in valid) - margin)
    y0 = max(0, min(box[1] for box in valid) - margin)
    x1 = min(width, max(box[2] for box in valid) + margin)
    y1 = min(height, max(box[3] for box in valid) + margin)
    if x1 - x0 > max_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - max_side // 2)
        x1 = min(width, x0 + max_side)
        x0 = max(0, x1 - max_side)
    if y1 - y0 > max_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - max_side // 2)
        y1 = min(height, y0 + max_side)
        y0 = max(0, y1 - max_side)
    return x0, y0, x1, y1


def normalize_heat(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    colors = np.zeros(values.shape + (3,), dtype=np.uint8)
    colors[..., 0] = np.asarray(255 * scaled, dtype=np.uint8)
    colors[..., 1] = np.asarray(255 * (1.0 - np.abs(scaled - 0.5) * 2.0), dtype=np.uint8)
    colors[..., 2] = np.asarray(255 * (1.0 - scaled), dtype=np.uint8)
    return colors


def render_mask_panel(segments_crop: np.ndarray, mask_by_segment: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    out = np.full(segments_crop.shape + (3,), 238, dtype=np.uint8)
    valid = segments_crop >= 0
    hit = valid & mask_by_segment[np.maximum(segments_crop, 0)]
    out[hit] = color
    out[valid & ~hit] = [210, 210, 210]
    return Image.fromarray(out, mode="RGB")


def render_score_panel(segments_crop: np.ndarray, scores: np.ndarray) -> Image.Image:
    values = np.zeros(segments_crop.shape, dtype=np.float32)
    valid = segments_crop >= 0
    values[valid] = scores[np.maximum(segments_crop[valid], 0)]
    colors = normalize_heat(values)
    colors[~valid] = [245, 245, 245]
    return Image.fromarray(colors, mode="RGB")


def add_header(image: Image.Image, title: str) -> Image.Image:
    header = 34
    out = Image.new("RGB", (image.width, image.height + header), (245, 245, 245))
    out.paste(image, (0, header))
    ImageDraw.Draw(out).text((8, 9), title, fill=(20, 20, 20))
    return out


def resize_panel(image: Image.Image, max_dim: int) -> Image.Image:
    scale = min(1.0, max_dim / max(image.width, image.height))
    if scale >= 1.0:
        return image
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)


def make_grid(panels: list[Image.Image], cols: int = 2, pad: int = 8) -> Image.Image:
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / cols))
    out = Image.new("RGB", (cols * width + (cols + 1) * pad, rows * height + (rows + 1) * pad), (255, 255, 255))
    for index, panel in enumerate(panels):
        x = pad + (index % cols) * (width + pad)
        y = pad + (index // cols) * (height + pad)
        out.paste(panel, (x, y))
    return out


def visual_one(
    row: dict[str, str],
    prompt: dict[str, str],
    rank: int,
    out_dir: Path,
    crop_margin: int,
    crop_max_side: int,
    panel_max_dim: int,
) -> str:
    image_id = row["image_id"]
    scale = row["scale"]
    scale_dir = MULTISCALE_ROOT / image_id / scale
    segments = np.load(scale_dir / "superpixels.npy", mmap_mode="r")
    he = np.load(scale_dir / "he_10x_rgb.npy", mmap_mode="r")
    score_path = SCORE_DIR / f"{row['query_id']}_{scale}.npz"
    score = np.load(score_path)
    hard = score["gt_hard_label_per_superpixel"].astype(bool)
    scores = score["score_final"].astype(np.float32)
    areas = score["area"].astype(np.float64)
    pred_area = float(row.get("PredArea") or 0.0)
    pred_fraction = pred_area / max(float(np.sum(areas)), 1.0)
    pred = predict_top_area(scores, areas, float(np.clip(pred_fraction, 0.001, 0.95)))
    prompt_boxes = parse_boxes(prompt.get("positive_boxes", "")) + parse_boxes(prompt.get("negative_boxes", ""))
    pred_box = target_bbox(np.asarray(segments), pred)
    gt_box = target_bbox(np.asarray(segments), hard)
    x0, y0, x1, y1 = union_crop(prompt_boxes + [box for box in (pred_box, gt_box) if box], segments.shape, crop_margin, crop_max_side)
    he_crop = np.asarray(he[y0:y1, x0:x1, :3]).astype(np.uint8)
    seg_crop = np.asarray(segments[y0:y1, x0:x1])

    he_panel = Image.fromarray(he_crop, mode="RGB")
    draw = ImageDraw.Draw(he_panel)
    for box in parse_boxes(prompt.get("positive_boxes", "")):
        draw.rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), outline=(0, 255, 0), width=5)
    for box in parse_boxes(prompt.get("negative_boxes", "")):
        draw.rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), outline=(255, 0, 0), width=5)

    panels = [
        add_header(resize_panel(he_panel, panel_max_dim), f"HE + prompts | rank {rank} | {scale}"),
        add_header(resize_panel(render_mask_panel(seg_crop, hard, (40, 170, 70)), panel_max_dim), "GT target"),
        add_header(resize_panel(render_mask_panel(seg_crop, pred, (230, 0, 140)), panel_max_dim), "Prediction classwise toparea"),
        add_header(resize_panel(render_score_panel(seg_crop, scores), panel_max_dim), "score_final heatmap"),
    ]
    grid = make_grid(panels)
    out_name = f"{rank:05d}_{safe_name(row['query_id'])}_{scale}.png"
    out_path = out_dir / out_name
    grid.save(out_path)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottom_fraction", type=float, default=0.20)
    parser.add_argument("--max_cases", type=int, default=0, help="0 means all bottom-fraction cases.")
    parser.add_argument("--crop_margin", type=int, default=1200)
    parser.add_argument("--crop_max_side", type=int, default=4096)
    parser.add_argument("--panel_max_dim", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [row for row in read_csv(METRICS_PATH) if row.get("status") == "ok"]
    metrics.sort(key=lambda row: (float(row["Dice_classwise_toparea"]), float(row["BestDice"]), row["query_id"], row["scale"]))
    bottom_count = max(1, int(len(metrics) * args.bottom_fraction))
    bottom = metrics[:bottom_count]
    if args.max_cases > 0:
        bottom = bottom[: args.max_cases]
    prompts = {row["query_id"]: row for row in read_csv(PROMPT_PATH)}
    out_dir = VIS_ROOT / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for rank, row in enumerate(bottom, start=1):
        prompt = prompts[row["query_id"]]
        visual_path = visual_one(row, prompt, rank, out_dir, args.crop_margin, args.crop_max_side, args.panel_max_dim)
        index_rows.append(
            {
                "rank": rank,
                "query_id": row["query_id"],
                "image_id": row["image_id"],
                "target_class": row["target_class"],
                "scale": row["scale"],
                "prompt_quality": row["prompt_quality"],
                "prompt_mode": row["prompt_mode"],
                "Dice_classwise_toparea": row["Dice_classwise_toparea"],
                "BestDice": row["BestDice"],
                "mAP": row["mAP"],
                "AUROC": row["AUROC"],
                "PredArea": row["PredArea"],
                "GTArea": row["GTArea"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "status": row["status"],
                "visual_path": visual_path,
            }
        )
        if rank % 100 == 0:
            print(f"visualized bottom20 {rank}/{len(bottom)}", flush=True)
    write_csv(VIS_ROOT / "bottom20_index.csv", index_rows, FIELDS)
    summary = {
        "source_metrics": str(METRICS_PATH),
        "bottom_fraction": args.bottom_fraction,
        "available_ok_rows": len(metrics),
        "bottom20_total_rows": bottom_count,
        "visualized_rows": len(index_rows),
        "dice_threshold": float(bottom[-1]["Dice_classwise_toparea"]) if bottom else None,
        "scale_counts": dict(Counter(row["scale"] for row in bottom)),
        "quality_counts": dict(Counter(row["prompt_quality"] for row in bottom)),
        "index_csv": str(VIS_ROOT / "bottom20_index.csv"),
        "case_dir": str(out_dir),
    }
    (VIS_ROOT / "bottom20_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase C Bottom 20% Visualization",
        "",
        f"- source metrics: {METRICS_PATH}",
        f"- bottom fraction: {args.bottom_fraction}",
        f"- available ok rows: {len(metrics)}",
        f"- bottom20 total rows: {bottom_count}",
        f"- visualized rows: {len(index_rows)}",
        f"- dice threshold: {summary['dice_threshold']}",
        f"- index: {VIS_ROOT / 'bottom20_index.csv'}",
        f"- cases: {out_dir}",
        "",
        f"scale_counts: `{summary['scale_counts']}`",
        f"quality_counts: `{summary['quality_counts']}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
