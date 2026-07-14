# 作用：将当前 Phase C 选中的完整 superpixel 按原始 GT 像素逐像素评估 Dice 和 mIoU。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_c.run_multiscale_baseline import (
    FRACTIONS,
    METRICS_PATH,
    MULTISCALE_ROOT,
    NUM_CLASSES,
    SCORE_DIR,
    class_area_priors,
    compute_gt_counts,
    load_gt_mask,
    read_csv,
)

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST = PRET_ROOT / "data_manifest_v3.csv"
OUT = PRET_ROOT / "evaluations" / "multiscale_pixel_metrics.csv"
SUMMARY = PRET_ROOT / "evaluations" / "pixel_class_scale_summary.csv"
REPORT = Path(__file__).resolve().parent / "pixel_metrics_report.md"
SCALES = ("small", "medium", "large")

FIELDS = [
    "query_id", "image_id", "target_class", "scale", "prompt_quality", "prompt_mode",
    "PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice", "PixelBestArea",
    "PixelPrecision", "PixelRecall", "PixelTP", "PixelFP", "PixelFN", "PixelPredArea",
    "PixelGTArea", "PixelEvalArea", "status",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def pixel_metrics_from_prefix(
    cumulative_target: np.ndarray,
    cumulative_pred: np.ndarray,
    prefix_size: int,
    gt_area: float,
) -> dict[str, float]:
    """Exactly equals rasterizing the first `prefix_size` score-ranked superpixels."""
    tp = float(cumulative_target[prefix_size - 1]) if prefix_size else 0.0
    pred_area = float(cumulative_pred[prefix_size - 1]) if prefix_size else 0.0
    fp = pred_area - tp
    fn = gt_area - tp
    denominator = 2.0 * tp + fp + fn
    dice = 2.0 * tp / denominator if denominator else 1.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return {
        "dice": dice,
        "iou": iou,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def prefix_size_for_fraction(cumulative_area: np.ndarray, fraction: float) -> int:
    limit = float(cumulative_area[-1]) * fraction
    # Mirrors predict_top_area: the superpixel that first reaches the limit is included.
    return min(int(np.searchsorted(cumulative_area, limit, side="left")) + 1, len(cumulative_area))


def load_counts(image_id: str, scale: str, gt_path: str, cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (image_id, scale)
    if key not in cache:
        directory = MULTISCALE_ROOT / image_id / scale
        segments = np.load(directory / "superpixels.npy", mmap_mode="r")
        gt = load_gt_mask(Path(gt_path))
        if gt.shape != segments.shape:
            gt = np.asarray(
                Image.fromarray(gt.astype(np.uint8)).resize(
                    (segments.shape[1], segments.shape[0]), Image.Resampling.NEAREST
                ),
                dtype=np.int16,
            )
        counts, valid_pixels = compute_gt_counts(segments, gt, int(segments.max()) + 1)
        areas = np.asarray([float(row.get("area", 0.0)) for row in read_csv(directory / "superpixels.csv")], dtype=np.float64)
        cache[key] = counts, valid_pixels, areas
    return cache[key]


def evaluate_image(
    image_id: str,
    rows: list[dict[str, str]],
    gt_path: str,
    priors: dict[tuple[str, int, str], tuple[float, float]],
) -> list[dict[str, object]]:
    """Evaluate all query-scale rows for one image so GT counts are loaded only once per scale."""
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    output: list[dict[str, object]] = []
    for row in rows:
        item = {key: row.get(key, "") for key in ("query_id", "image_id", "target_class", "scale", "prompt_quality", "prompt_mode")}
        if row.get("status") != "ok":
            output.append({**item, "status": row.get("status", "")})
            continue
        scale, class_id = row["scale"], int(row["target_class"])
        counts, valid_pixels, areas = load_counts(image_id, scale, gt_path, cache)
        candidate = valid_pixels > 0
        candidate_ids = np.flatnonzero(candidate)
        with np.load(SCORE_DIR / f"{row['query_id']}_{scale}.npz") as score_file:
            scores = score_file["score_final"].astype(np.float64)[candidate]
        local_areas = areas[candidate]
        order = np.argsort(-scores, kind="stable")
        ordered_ids = candidate_ids[order]
        cumulative_area = np.cumsum(local_areas[order])
        cumulative_target = np.cumsum(counts[ordered_ids, class_id], dtype=np.float64)
        cumulative_pred = np.cumsum(valid_pixels[ordered_ids], dtype=np.float64)
        gt_area = float(np.sum(counts[:, class_id]))
        _, fraction = priors[(image_id, class_id, scale)]
        primary = pixel_metrics_from_prefix(
            cumulative_target, cumulative_pred, prefix_size_for_fraction(cumulative_area, fraction), gt_area,
        )
        best_dice, best_area = -1.0, 0.0
        for value in FRACTIONS:
            dice = pixel_metrics_from_prefix(
                cumulative_target, cumulative_pred, prefix_size_for_fraction(cumulative_area, value), gt_area,
            )["dice"]
            if dice > best_dice:
                best_dice, best_area = dice, value
        output.append({
            **item,
            "PixelDice_classwise_toparea": primary["dice"], "Pixel_mIoU": primary["iou"],
            "PixelBestDice": best_dice, "PixelBestArea": best_area,
            "PixelPrecision": primary["precision"], "PixelRecall": primary["recall"],
            "PixelTP": primary["tp"], "PixelFP": primary["fp"], "PixelFN": primary["fn"],
            "PixelPredArea": primary["pred_area"], "PixelGTArea": primary["gt_area"],
            "PixelEvalArea": float(np.sum(valid_pixels)), "status": "ok",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug limit; 0 evaluates all metric rows.")
    parser.add_argument("--workers", type=int, default=1, help="按 image_id 并行的 worker 数量。")
    args = parser.parse_args()
    manifest_rows = read_csv(MANIFEST)
    manifest = {row["image_id"]: row for row in manifest_rows}
    priors = class_area_priors(manifest_rows)
    base_rows = read_csv(METRICS_PATH)
    if args.max_rows:
        base_rows = base_rows[:args.max_rows]
    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in base_rows:
        rows_by_image[row["image_id"]].append(row)
    output: list[dict[str, object]] = []
    jobs = [(image_id, rows_by_image[image_id], manifest[image_id]["gt_mask_path"], priors) for image_id in sorted(rows_by_image)]
    if args.workers <= 1:
        for index, job in enumerate(jobs, start=1):
            output.extend(evaluate_image(*job))
            print(f"pixel_metrics {index}/{len(jobs)} image_id={job[0]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_image, *job): job[0] for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                output.extend(future.result())
                print(f"pixel_metrics {index}/{len(jobs)} image_id={futures[future]}", flush=True)
    output.sort(key=lambda row: (str(row["image_id"]), str(row["query_id"]), str(row["scale"])))

    write_csv(OUT, output, FIELDS)
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in output:
        if row["status"] == "ok":
            groups[(int(row["target_class"]), str(row["scale"]))].append(row)
    summary = []
    measure_keys = ("PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice", "PixelPrecision", "PixelRecall")
    for (class_id, scale), rows in sorted(groups.items()):
        summary.append({"target_class": class_id, "scale": scale, "n": len(rows), **{
            key: float(np.mean([float(row[key]) for row in rows])) for key in measure_keys
        }})
    write_csv(SUMMARY, summary, ["target_class", "scale", "n", *measure_keys])
    ok = [row for row in output if row["status"] == "ok"]
    overall = {key: float(np.mean([float(row[key]) for row in ok])) for key in measure_keys}
    REPORT.write_text(
        "# Phase C Pixel Metrics\n\n"
        "- prediction: selected full superpixels from the formal LOIO classwise area prior.\n"
        "- GT: original raw GT pixels; only annotated pixels covered by a superpixel are evaluated.\n"
        "- computation: segment-wise pixel counts, mathematically identical to rasterizing masks.\n"
        f"- overall: {json.dumps(overall)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "ok": len(ok), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
