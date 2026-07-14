# 作用：用原始 GT 中“superpixel 含任意 target 像素即为正”的口径重算三尺度 retrieval 指标。

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import binary_segmentation_metrics, predict_top_area
from benchmarks.v3.phase_c.run_multiscale_baseline import (
    METRICS_PATH, MULTISCALE_ROOT, NUM_CLASSES, SCORE_DIR, compute_gt_counts,
    load_gt_mask, ranking_metrics, read_csv,
)

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST = PRET_ROOT / "data_manifest_v3.csv"
OUT = PRET_ROOT / "evaluations" / "multiscale_gt_presence_metrics.csv"
SUMMARY = PRET_ROOT / "evaluations" / "gt_presence_class_scale_summary.csv"
REPORT = Path(__file__).resolve().parent / "gt_presence_metrics_report.md"
SCALES = ("small", "medium", "large")
FIELDS = ["query_id", "image_id", "target_class", "scale", "prompt_quality", "prompt_mode", "mAP_gt_presence", "AUROC_gt_presence", "Dice_gt_presence", "mIoU_gt_presence", "BestDice_gt_presence", "Precision_gt_presence", "Recall_gt_presence", "PredArea_gt_presence", "GTArea_gt_presence", "PresenceAreaFraction", "status"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def presence_data(image_id: str, scale: str, gt_path: str, cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (image_id, scale)
    if key not in cache:
        directory = MULTISCALE_ROOT / image_id / scale
        segments = np.load(directory / "superpixels.npy", mmap_mode="r")
        gt = load_gt_mask(Path(gt_path))
        if gt.shape != segments.shape:
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((segments.shape[1], segments.shape[0]), Image.Resampling.NEAREST), dtype=np.int16)
        count, valid = compute_gt_counts(segments, gt, int(segments.max()) + 1)
        areas = np.asarray([float(row.get("area", 0)) for row in read_csv(directory / "superpixels.csv")], dtype=np.float64)
        cache[key] = count, valid > 0, areas
    return cache[key]


def build_priors(manifest_rows: list[dict[str, str]], cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[tuple[str, int, str], float]:
    by_class_scale: dict[tuple[int, str], list[tuple[str, float]]] = defaultdict(list)
    for manifest in manifest_rows:
        for scale in SCALES:
            counts, valid, areas = presence_data(manifest["image_id"], scale, manifest["gt_mask_path"], cache)
            total = areas[valid].sum()
            for class_id in range(NUM_CLASSES):
                target = counts[:, class_id] > 0
                by_class_scale[(class_id, scale)].append((manifest["image_id"], float(areas[valid & target].sum() / max(total, 1.0))))
    priors: dict[tuple[str, int, str], float] = {}
    for (class_id, scale), values in by_class_scale.items():
        fallback = float(np.median([value for _, value in values]))
        for image_id, _ in values:
            train = [value for other, value in values if other != image_id]
            priors[(image_id, class_id, scale)] = float(np.clip(np.median(train) if train else fallback, 0.001, 0.95))
    return priors


def main() -> None:
    manifest_rows = read_csv(MANIFEST)
    manifest = {row["image_id"]: row for row in manifest_rows}
    base_rows = read_csv(METRICS_PATH)
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    priors = build_priors(manifest_rows, cache)
    output: list[dict[str, object]] = []
    for index, row in enumerate(base_rows, start=1):
        item = {key: row.get(key, "") for key in ("query_id", "image_id", "target_class", "scale", "prompt_quality", "prompt_mode")}
        if row.get("status") != "ok":
            output.append({**item, "status": row.get("status", "")})
            continue
        image_id, scale, class_id = row["image_id"], row["scale"], int(row["target_class"])
        counts, valid, areas = presence_data(image_id, scale, manifest[image_id]["gt_mask_path"], cache)
        with np.load(SCORE_DIR / f"{row['query_id']}_{scale}.npz") as score_file:
            scores = score_file["score_final"].astype(np.float64)[valid]
        local_areas = areas[valid]
        target = (counts[:, class_id] > 0)[valid]
        fraction = priors[(image_id, class_id, scale)]
        pred = predict_top_area(scores, local_areas, fraction)
        metric = binary_segmentation_metrics(target, pred)
        ap, auc = ranking_metrics(target, scores)
        best = max(float(binary_segmentation_metrics(target, predict_top_area(scores, local_areas, value))["dice"]) for value in np.linspace(0.001, 0.95, 96))
        output.append({**item, "mAP_gt_presence": ap, "AUROC_gt_presence": auc, "Dice_gt_presence": float(metric["dice"]), "mIoU_gt_presence": float(metric["iou"]), "BestDice_gt_presence": best, "Precision_gt_presence": float(metric["precision"]), "Recall_gt_presence": float(metric["recall"]), "PredArea_gt_presence": float(local_areas[pred].sum()), "GTArea_gt_presence": float(local_areas[target].sum()), "PresenceAreaFraction": fraction, "status": "ok"})
        if index % 500 == 0:
            print(f"gt_presence {index}/{len(base_rows)}", flush=True)
    write_csv(OUT, output)
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in output:
        if row["status"] == "ok":
            groups[(int(row["target_class"]), str(row["scale"]))].append(row)
    summary = []
    for (class_id, scale), rows in sorted(groups.items()):
        summary.append({"target_class": class_id, "scale": scale, "n": len(rows), **{key: float(np.mean([float(row[key]) for row in rows])) for key in ("mAP_gt_presence", "AUROC_gt_presence", "Dice_gt_presence", "mIoU_gt_presence", "BestDice_gt_presence", "Precision_gt_presence", "Recall_gt_presence", "PresenceAreaFraction")}})
    write_csv(SUMMARY, summary)
    ok = [row for row in output if row["status"] == "ok"]
    overall = {key: float(np.mean([float(row[key]) for row in ok])) for key in ("mAP_gt_presence", "AUROC_gt_presence", "Dice_gt_presence", "mIoU_gt_presence", "BestDice_gt_presence")}
    REPORT.write_text("# Phase C GT-presence Metrics\n\n" + "- definition: a superpixel is target-positive if raw GT contains at least one pixel of the target class.\n" + "- overall: " + json.dumps(overall) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "ok": len(ok), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
