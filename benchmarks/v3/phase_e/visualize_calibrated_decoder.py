# 作用：按类别可视化 calibrated MLP decoder 的最好、中间、最差 OOF 案例，并与 Phase C baseline 和原始像素 GT 对比。

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_visualize import CLASS_NAMES, boundaries_overlay, save_single_panel, segment_values_to_image
from benchmarks.v3.phase_c.run_multiscale_baseline import MULTISCALE_ROOT, load_gt_mask, read_csv
from benchmarks.v3.phase_c.visualize_top_bottom_dice_full_grid import (
    _resize_arrays, predicted_mask, resize_raw_gt,
)
from benchmarks.v3.phase_e.train_mlp_mask_decoder import Decoder, ImageData, MANIFEST, METRICS, SCORE_DIR, TOKEN_FILE, node_features, predict
from benchmarks.v3.phase_e.train_mlp_mask_decoder_calibrated import MODELS_OUT, PREDICTIONS_OUT

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
VIS_ROOT = PRET_ROOT / "visualizations" / "phase_e_calibrated_decoder"
INDEX_OUT = VIS_ROOT / "visualization_index.csv"
SCALE = "small"
PANEL_NAMES = (
    "01_he_boundary.png", "02_prompt_overlay.png", "03_raw_gt.png", "04_phase_c_score.png",
    "05_phase_c_baseline.png", "06_mlp_probability.png", "07_calibrated_mask.png", "08_pixel_error.png",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_cases(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["target_class"])].append(row)
    selected = []
    for class_id, items in sorted(by_class.items()):
        items.sort(key=lambda row: float(row["Classwise_PixelDice"]))
        choices = (("worst", items[0]), ("middle", items[len(items) // 2]), ("best", items[-1]))
        selected.extend(choices)
    return selected


def load_visual_image(image_id: str, cache: dict[str, ImageData]) -> ImageData:
    if image_id in cache:
        return cache[image_id]
    directory = MULTISCALE_ROOT / image_id / SCALE
    rows = read_csv(directory / "superpixels.csv")
    edges = np.asarray(np.load(directory / "adjacency.npy", mmap_mode="r"), dtype=np.int64)
    edges = edges[(edges[:, 0] >= 0) & (edges[:, 0] < len(rows)) & (edges[:, 1] >= 0) & (edges[:, 1] < len(rows))]
    valid = np.asarray([float(row.get("valid_fraction", 0.0)) > 0 for row in rows], dtype=np.float64)
    data = ImageData(
        tokens=np.asarray(np.load(directory / TOKEN_FILE, mmap_mode="r"), dtype=np.float32),
        areas=np.asarray([float(row["area"]) for row in rows], dtype=np.float32),
        centers=np.asarray([[float(row["center_x"]), float(row["center_y"])] for row in rows], dtype=np.float32),
        density=np.asarray([float(row.get("cell_density", 0.0)) for row in rows], dtype=np.float32),
        count=np.asarray([float(row.get("cell_count", 0.0)) for row in rows], dtype=np.float32),
        valid_pixels=valid,
        counts=np.zeros((len(rows), 12), dtype=np.float64),
        edges=edges,
    )
    cache[image_id] = data
    return data


def mask_panel(segments: np.ndarray, selected: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    colors = np.full((len(selected), 3), [230, 230, 230], dtype=np.uint8)
    colors[selected] = color
    return segment_values_to_image(segments, np.zeros(len(selected), dtype=np.float32), colors)


def prompt_panel(segments: np.ndarray, count: int, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    colors = np.full((count, 3), [230, 230, 230], dtype=np.uint8)
    positive = positive[(positive >= 0) & (positive < count)]
    negative = negative[(negative >= 0) & (negative < count)]
    colors[positive] = [0, 80, 255]
    colors[negative] = [230, 0, 140]
    return segment_values_to_image(segments, np.zeros(count, dtype=np.float32), colors)


def pixel_error(render_segments: np.ndarray, render_gt: np.ndarray, class_id: int, selected: np.ndarray) -> np.ndarray:
    predicted = np.zeros(render_segments.shape, dtype=bool)
    covered = render_segments >= 0
    predicted[covered] = selected[render_segments[covered]]
    valid = covered & (render_gt >= 0) & (render_gt < 12)
    target = render_gt == class_id
    output = np.full((*render_segments.shape, 3), [235, 235, 235], dtype=np.uint8)
    output[valid & predicted & target] = [20, 170, 60]
    output[valid & predicted & ~target] = [230, 0, 140]
    output[valid & ~predicted & target] = [80, 80, 80]
    return output


def combine(case_dir: Path) -> Path:
    panels = [Image.open(case_dir / name).convert("RGB") for name in PANEL_NAMES]
    width = 760
    resized = [panel.resize((width, max(1, round(panel.height * width / panel.width))), Image.Resampling.BILINEAR) for panel in panels]
    row_heights = [max(image.height for image in resized[:4]), max(image.height for image in resized[4:])]
    canvas = Image.new("RGB", (4 * width, sum(row_heights)), "white")
    for index, image in enumerate(resized):
        row, column = divmod(index, 4)
        canvas.paste(image, (column * width, 0 if row == 0 else row_heights[0]))
    output = case_dir / "00_combined_grid.png"
    canvas.save(output)
    for image in panels + resized:
        image.close()
    canvas.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_render_dimension", type=int, default=3072)
    parser.add_argument("--keep_panels", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("visualizer requires CUDA for OOF decoder inference")
    device = torch.device("cuda")
    predictions = read_csv(PREDICTIONS_OUT)
    metrics = {(row["query_id"], row["scale"]): row for row in read_csv(METRICS)}
    manifest_rows = read_csv(MANIFEST)
    manifest = {row["image_id"]: row for row in manifest_rows}
    source = torch.load(MODELS_OUT, map_location=device, weights_only=False)
    checkpoints = {int(item["test_fold"]): item for item in source["fold_checkpoints"]}
    image_cache = {}
    index_rows = []

    for bucket, row in select_cases(predictions):
        fold, image_id = int(row["fold"]), row["image_id"]
        class_id, query_id = int(row["target_class"]), row["query_id"]
        checkpoint = checkpoints[fold]
        model = Decoder(int(checkpoint["feature_dim"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        data = load_visual_image(image_id, image_cache)
        features, valid = node_features(data, SCORE_DIR / f"{query_id}_{SCALE}.npz")
        probability = predict(model, features, checkpoint["feature_mean"], checkpoint["feature_std"], device)
        calibrated = (probability >= float(row["classwise_threshold"])) & valid

        scale_dir = MULTISCALE_ROOT / image_id / SCALE
        rgb = np.load(scale_dir / "he_10x_rgb.npy", mmap_mode="r")
        segments = np.load(scale_dir / "superpixels.npy", mmap_mode="r")
        render_rgb, render_segments = _resize_arrays(rgb, segments, args.max_render_dimension)
        raw_gt_rgb = resize_raw_gt(Path(manifest[image_id]["gt_mask_path"]), render_segments.shape)
        gt = load_gt_mask(Path(manifest[image_id]["gt_mask_path"]))
        if gt.shape != segments.shape:
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((segments.shape[1], segments.shape[0]), Image.Resampling.NEAREST), dtype=np.int16)
        if gt.shape != render_segments.shape:
            render_gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((render_segments.shape[1], render_segments.shape[0]), Image.Resampling.NEAREST), dtype=np.int16)
        else:
            render_gt = gt

        with np.load(SCORE_DIR / f"{query_id}_{SCALE}.npz") as score:
            scores = score["score_final"].astype(np.float32)
            score_rank = score["rank_percentile"].astype(np.float32)
            positive = score["positive_prompt_segments"].astype(np.int64)
            negative = score["negative_prompt_segments"].astype(np.int64)
            baseline = predicted_mask(metrics[(query_id, SCALE)], score)

        case_dir = VIS_ROOT / f"class_{class_id:02d}" / bucket / f"{safe_name(query_id)}_dice{float(row['Classwise_PixelDice']):.3f}"
        case_dir.mkdir(parents=True, exist_ok=True)
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        common = [
            f"query={query_id} | image={image_id} | fold={fold} | bucket={bucket}",
            f"class={class_id} {class_name} | quality={row['prompt_quality']} | threshold={float(row['classwise_threshold']):.2f}",
            f"calibrated Dice={float(row['Classwise_PixelDice']):.3f} mIoU={float(row['Classwise_Pixel_mIoU']):.3f}",
            f"baseline Dice={float(row['Baseline_PixelDice']):.3f} | calibrated Precision={float(row['Classwise_PixelPrecision']):.3f} Recall={float(row['Classwise_PixelRecall']):.3f}",
            f"raw score p02/median/p98={np.quantile(scores, 0.02):.3f}/{np.median(scores):.3f}/{np.quantile(scores, 0.98):.3f}",
        ]
        save_single_panel(case_dir / PANEL_NAMES[0], "01 HE + small superpixel boundary", boundaries_overlay(render_rgb, render_segments), common)
        save_single_panel(case_dir / PANEL_NAMES[1], "02 Prompt overlay", prompt_panel(render_segments, len(probability), positive, negative), common + ["blue=positive; magenta=negative"])
        save_single_panel(case_dir / PANEL_NAMES[2], "03 Raw GT tissue mask", raw_gt_rgb, common)
        save_single_panel(case_dir / PANEL_NAMES[3], "04 Phase C score percentile rank", segment_values_to_image(render_segments, score_rank), common + ["rank view reveals relative ordering when raw cosine is saturated"], add_heat_legend=True)
        save_single_panel(case_dir / PANEL_NAMES[4], "05 Phase C fixed-small mask", mask_panel(render_segments, baseline, (20, 180, 70)), common)
        save_single_panel(case_dir / PANEL_NAMES[5], "06 MLP target probability", segment_values_to_image(render_segments, probability), common, add_heat_legend=True)
        save_single_panel(case_dir / PANEL_NAMES[6], "07 Classwise-calibrated mask", mask_panel(render_segments, calibrated, (20, 180, 70)), common)
        save_single_panel(case_dir / PANEL_NAMES[7], "08 Raw-pixel error map", pixel_error(render_segments, render_gt, class_id, calibrated), common + ["green=TP; magenta=FP; dark gray=FN"])
        combined = combine(case_dir)
        if not args.keep_panels:
            for name in PANEL_NAMES:
                (case_dir / name).unlink()
        summary = {**row, "bucket": bucket, "combined_grid": str(combined)}
        (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_rows.append({"target_class": class_id, "class_name": class_name, "bucket": bucket, "query_id": query_id, "image_id": image_id, "fold": fold, "Classwise_PixelDice": row["Classwise_PixelDice"], "Baseline_PixelDice": row["Baseline_PixelDice"], "classwise_threshold": row["classwise_threshold"], "path": str(combined)})
        print(f"decoder_visualization class={class_id} bucket={bucket} query={query_id}", flush=True)
    write_csv(INDEX_OUT, index_rows)
    print(json.dumps({"cases": len(index_rows), "output": str(VIS_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
