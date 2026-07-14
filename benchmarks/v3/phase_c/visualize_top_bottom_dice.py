# 作用：可视化 Phase C 中 Dice_classwise_toparea 最高 50 和最低 50 的 query-scale 案例。

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
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
VIS_ROOT = PRET_ROOT / "visualizations" / "phase_c_dice_top_bottom"
REPORT_PATH = Path(__file__).resolve().parent / "dice_top_bottom_visual_report.md"

INDEX_FIELDS = [
    "group",
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
    "Precision",
    "Recall",
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    boxes = []
    for item in str(text or "").split(";"):
        if not item:
            continue
        boxes.append(tuple(int(float(value)) for value in item.split(":")))
    return boxes


def segment_bbox(segments: np.ndarray, mask_by_segment: np.ndarray, max_segments: int = 120) -> tuple[int, int, int, int] | None:
    ids = np.flatnonzero(mask_by_segment)
    if ids.size == 0:
        return None
    ids = ids[:max_segments]
    mask = np.isin(segments, ids)
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def union_crop(
    boxes: list[tuple[int, int, int, int] | None],
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


def segment_mask(segments_crop: np.ndarray, mask_by_segment: np.ndarray) -> np.ndarray:
    valid = segments_crop >= 0
    out = np.zeros(segments_crop.shape, dtype=bool)
    out[valid] = mask_by_segment[np.maximum(segments_crop[valid], 0)]
    return out


def overlay_color(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base.astype(np.float32, copy=True)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_prompt_boxes(draw: ImageDraw.ImageDraw, prompt: dict[str, str], x0: int, y0: int) -> None:
    for box in parse_boxes(prompt.get("positive_boxes", "")):
        draw.rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), outline=(0, 255, 0), width=5)
    for box in parse_boxes(prompt.get("negative_boxes", "")):
        draw.rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), outline=(255, 0, 0), width=5)


def normalize_heat(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    colors = np.zeros(values.shape + (3,), dtype=np.uint8)
    colors[..., 0] = np.asarray(255 * scaled, dtype=np.uint8)
    colors[..., 1] = np.asarray(220 * (1.0 - np.abs(scaled - 0.5) * 2.0), dtype=np.uint8)
    colors[..., 2] = np.asarray(255 * (1.0 - scaled), dtype=np.uint8)
    return colors


def score_heatmap(segments_crop: np.ndarray, scores: np.ndarray) -> Image.Image:
    values = np.zeros(segments_crop.shape, dtype=np.float32)
    valid = segments_crop >= 0
    values[valid] = scores[np.maximum(segments_crop[valid], 0)]
    colors = normalize_heat(values)
    colors[~valid] = [245, 245, 245]
    return Image.fromarray(colors, mode="RGB")


def resize_panel(image: Image.Image, max_dim: int) -> Image.Image:
    scale = min(1.0, max_dim / max(image.width, image.height))
    if scale >= 1.0:
        return image
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)


def add_header(image: Image.Image, title: str) -> Image.Image:
    header_h = 42
    out = Image.new("RGB", (image.width, image.height + header_h), (245, 245, 245))
    out.paste(image, (0, header_h))
    ImageDraw.Draw(out).text((8, 12), title, fill=(15, 15, 15))
    return out


def grid(panels: list[Image.Image], cols: int = 2, pad: int = 8) -> Image.Image:
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / cols))
    out = Image.new("RGB", (cols * width + (cols + 1) * pad, rows * height + (rows + 1) * pad), (255, 255, 255))
    for idx, panel in enumerate(panels):
        x = pad + (idx % cols) * (width + pad)
        y = pad + (idx // cols) * (height + pad)
        out.paste(panel, (x, y))
    return out


def make_visual(
    row: dict[str, str],
    prompt: dict[str, str],
    group: str,
    rank: int,
    out_dir: Path,
    crop_margin: int,
    crop_max_side: int,
    panel_max_dim: int,
) -> str:
    image_id = row["image_id"]
    scale = row["scale"]
    scale_dir = MULTISCALE_ROOT / image_id / scale
    segments = np.asarray(np.load(scale_dir / "superpixels.npy", mmap_mode="r"))
    he = np.asarray(np.load(scale_dir / "he_10x_rgb.npy", mmap_mode="r"))
    score = np.load(SCORE_DIR / f"{row['query_id']}_{scale}.npz")
    hard = score["gt_hard_label_per_superpixel"].astype(bool)
    scores = score["score_final"].astype(np.float32)
    areas = score["area"].astype(np.float64)
    pred_fraction = float(row.get("PredArea") or 0.0) / max(float(np.sum(areas)), 1.0)
    pred = predict_top_area(scores, areas, float(np.clip(pred_fraction, 0.001, 0.95)))

    prompt_boxes = parse_boxes(prompt.get("positive_boxes", "")) + parse_boxes(prompt.get("negative_boxes", ""))
    crop = union_crop(
        prompt_boxes + [segment_bbox(segments, hard), segment_bbox(segments, pred)],
        segments.shape,
        crop_margin,
        crop_max_side,
    )
    x0, y0, x1, y1 = crop
    he_crop = he[y0:y1, x0:x1, :3].astype(np.uint8)
    seg_crop = segments[y0:y1, x0:x1]
    gt_mask = segment_mask(seg_crop, hard)
    pred_mask = segment_mask(seg_crop, pred)
    tp = gt_mask & pred_mask
    fp = ~gt_mask & pred_mask
    fn = gt_mask & ~pred_mask

    overlay = overlay_color(he_crop, gt_mask, (30, 190, 80), 0.30)
    overlay = overlay_color(overlay, pred_mask, (230, 0, 170), 0.30)
    overlay_img = Image.fromarray(overlay, mode="RGB")
    draw_prompt_boxes(ImageDraw.Draw(overlay_img), prompt, x0, y0)

    error = he_crop.copy()
    error = overlay_color(error, tp, (30, 190, 80), 0.45)
    error = overlay_color(error, fp, (230, 0, 170), 0.55)
    error = overlay_color(error, fn, (255, 210, 0), 0.55)
    error_img = Image.fromarray(error, mode="RGB")
    draw_prompt_boxes(ImageDraw.Draw(error_img), prompt, x0, y0)

    heat = score_heatmap(seg_crop, scores)
    draw_prompt_boxes(ImageDraw.Draw(heat), prompt, x0, y0)

    raw_img = Image.fromarray(he_crop, mode="RGB")
    draw_prompt_boxes(ImageDraw.Draw(raw_img), prompt, x0, y0)

    title = (
        f"{group} rank={rank} scale={scale} class={row['target_class']} "
        f"Dice={float(row['Dice_classwise_toparea']):.3f} Best={float(row['BestDice']):.3f} "
        f"mAP={float(row['mAP']):.3f}"
    )
    panels = [
        add_header(resize_panel(raw_img, panel_max_dim), "HE + prompts | green=positive box red=negative box"),
        add_header(resize_panel(overlay_img, panel_max_dim), "HE overlay | green=GT magenta=prediction"),
        add_header(resize_panel(error_img, panel_max_dim), "Error overlay | green=TP magenta=FP yellow=FN"),
        add_header(resize_panel(heat, panel_max_dim), "score_final heatmap | red=high blue=low"),
    ]
    out = grid(panels)
    header = Image.new("RGB", (out.width, out.height + 54), (245, 245, 245))
    header.paste(out, (0, 54))
    ImageDraw.Draw(header).text((10, 18), title, fill=(10, 10, 10))
    out_name = f"{group}_{rank:03d}_{safe_name(row['query_id'])}_{scale}.png"
    out_path = out_dir / group / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header.save(out_path)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--crop_margin", type=int, default=900)
    parser.add_argument("--crop_max_side", type=int, default=3200)
    parser.add_argument("--panel_max_dim", type=int, default=820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_csv(METRICS_PATH) if row.get("status") == "ok"]
    rows.sort(key=lambda row: float(row["Dice_classwise_toparea"]))
    bottom = rows[: args.count]
    top = list(reversed(rows[-args.count:]))
    prompts = {row["query_id"]: row for row in read_csv(PROMPT_PATH)}
    out_dir = VIS_ROOT / "cases"
    index_rows: list[dict[str, object]] = []
    for group, selected in (("bottom50", bottom), ("top50", top)):
        for rank, row in enumerate(selected, start=1):
            visual_path = make_visual(row, prompts[row["query_id"]], group, rank, out_dir, args.crop_margin, args.crop_max_side, args.panel_max_dim)
            index_rows.append(
                {
                    "group": group,
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
                    "Precision": row["Precision"],
                    "Recall": row["Recall"],
                    "visual_path": visual_path,
                }
            )
            print(f"visualized {group} {rank}/{len(selected)}", flush=True)
    write_csv(VIS_ROOT / "dice_top_bottom_index.csv", index_rows, INDEX_FIELDS)
    summary = {
        "metric": "Dice_classwise_toparea",
        "source_metrics": str(METRICS_PATH),
        "count_per_group": args.count,
        "top_min_dice": float(top[-1]["Dice_classwise_toparea"]) if top else None,
        "top_max_dice": float(top[0]["Dice_classwise_toparea"]) if top else None,
        "bottom_min_dice": float(bottom[0]["Dice_classwise_toparea"]) if bottom else None,
        "bottom_max_dice": float(bottom[-1]["Dice_classwise_toparea"]) if bottom else None,
        "top_scale_counts": dict(Counter(row["scale"] for row in top)),
        "bottom_scale_counts": dict(Counter(row["scale"] for row in bottom)),
        "index_csv": str(VIS_ROOT / "dice_top_bottom_index.csv"),
        "case_dir": str(out_dir),
    }
    (VIS_ROOT / "dice_top_bottom_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Phase C Dice Top/Bottom Visualization",
                "",
                f"- metric: `{summary['metric']}`",
                f"- top50 Dice range: {summary['top_min_dice']} - {summary['top_max_dice']}",
                f"- bottom50 Dice range: {summary['bottom_min_dice']} - {summary['bottom_max_dice']}",
                f"- index: {summary['index_csv']}",
                f"- cases: {summary['case_dir']}",
                f"- top scale counts: `{summary['top_scale_counts']}`",
                f"- bottom scale counts: `{summary['bottom_scale_counts']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
