# 作用：按 v2 classwise_toparea_loio 口径，用其他图的 BestArea 中位数重算 v3 Dice/mIoU。

from __future__ import annotations

import csv
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import binary_segmentation_metrics, predict_top_area
from benchmarks.v3.phase_c.run_multiscale_baseline import METRIC_FIELDS, METRICS_PATH, MULTISCALE_ROOT, NUM_CLASSES, SCORE_DIR, read_csv

OUT_METRICS = Path(__file__).resolve().parent / "v2style_bestarea_multiscale_metrics.csv"
OUT_SUMMARY = Path(__file__).resolve().parent / "v2style_bestarea_class_scale_summary.csv"
SCALES = ("small", "medium", "large")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def valid_mask(image_id: str, scale: str) -> np.ndarray:
    rows = read_csv(MULTISCALE_ROOT / image_id / scale / "superpixels.csv")
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    valid_fraction = np.asarray([float(row.get("valid_fraction", 0.0)) for row in rows], dtype=np.float32)
    return (labels >= 0) & (labels < NUM_CLASSES) & (valid_fraction > 0)


def loio_bestarea_priors(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], float]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_class_scale: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok_rows:
        by_class_scale[(int(row["target_class"]), row["scale"])].append(row)
    priors: dict[tuple[str, int, str], float] = {}
    for (class_id, scale), group in by_class_scale.items():
        fallback = float(np.median([float(row["BestArea"]) for row in group]))
        for row in group:
            image_id = row["image_id"]
            train = [float(item["BestArea"]) for item in group if item["image_id"] != image_id]
            prior = float(np.median(train)) if train else fallback
            priors[(image_id, class_id, scale)] = float(np.clip(prior, 0.01, 0.60))
    return priors


def recompute_row(row: dict[str, str], priors: dict[tuple[str, int, str], float], valid_cache: dict[tuple[str, str], np.ndarray]) -> dict[str, object]:
    out = dict(row)
    out["PreviousDice_classwise_toparea"] = row.get("Dice_classwise_toparea", "")
    out["PreviousmIoU"] = row.get("mIoU", "")
    if row.get("status") != "ok":
        out["V2StyleAreaFraction"] = ""
        return out
    image_id = row["image_id"]
    scale = row["scale"]
    class_id = int(row["target_class"])
    score_path = SCORE_DIR / f"{row['query_id']}_{scale}.npz"
    try:
        score_npz = np.load(score_path)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        out["status"] = "score_file_error"
        out["score_file_error"] = str(exc)
        out["V2StyleAreaFraction"] = ""
        return out
    valid = valid_cache.setdefault((image_id, scale), valid_mask(image_id, scale))
    scores = score_npz["score_final"].astype(np.float32)[valid]
    areas = score_npz["area"].astype(np.float64)[valid]
    target = score_npz["gt_hard_label_per_superpixel"].astype(bool)[valid]
    area_fraction = priors[(image_id, class_id, scale)]
    pred = predict_top_area(scores, areas, area_fraction)
    metrics = binary_segmentation_metrics(target, pred)
    pred_area = float(np.sum(areas[pred]))
    gt_area = float(np.sum(areas[target]))
    fp_area = float(np.sum(areas[pred & ~target]))
    fn_area = float(np.sum(areas[~pred & target]))
    out.update(
        {
            "Dice_classwise_toparea": float(metrics["dice"]),
            "mIoU": float(metrics["iou"]),
            "PredArea": pred_area,
            "GTArea": gt_area,
            "Precision": float(metrics["precision"]),
            "Recall": float(metrics["recall"]),
            "FP_area": fp_area,
            "FN_area": fn_area,
            "V2StyleAreaFraction": area_fraction,
            "gt_positive_segments": int(np.sum(target)),
            "gt_negative_segments": int(np.sum(~target)),
        }
    )
    return out


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    names = {
        0: "tumor_epithelium",
        1: "tumor_stroma",
        2: "background",
        3: "necrosis",
        4: "normal_gland",
        5: "normal_stroma",
        6: "submucosa_serosa",
        7: "muscle",
        8: "lymphocyte_aggregate",
        9: "mucus",
        10: "fat",
        11: "blood",
    }
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(int(row["target_class"]), str(row["scale"]))].append(row)
    output = []
    for class_id in range(NUM_CLASSES):
        for scale in SCALES:
            group = groups.get((class_id, scale), [])
            if not group:
                continue
            mean = lambda key: float(np.mean([float(row[key]) for row in group]))
            output.append(
                {
                    "target_class": class_id,
                    "class_name": names[class_id],
                    "scale": scale,
                    "n": len(group),
                    "dice_classwise_toparea": f"{mean('Dice_classwise_toparea'):.6f}",
                    "previous_dice": f"{mean('PreviousDice_classwise_toparea'):.6f}",
                    "delta_dice": f"{mean('Dice_classwise_toparea') - mean('PreviousDice_classwise_toparea'):.6f}",
                    "auc_auroc": f"{mean('AUROC'):.6f}",
                    "map": f"{mean('mAP'):.6f}",
                    "miou": f"{mean('mIoU'):.6f}",
                    "previous_miou": f"{mean('PreviousmIoU'):.6f}",
                    "delta_miou": f"{mean('mIoU') - mean('PreviousmIoU'):.6f}",
                    "precision": f"{mean('Precision'):.6f}",
                    "recall": f"{mean('Recall'):.6f}",
                    "area_fraction": f"{mean('V2StyleAreaFraction'):.6f}",
                    "best_dice": f"{mean('BestDice'):.6f}",
                }
            )
    return output


def main() -> None:
    metric_rows = read_csv(METRICS_PATH)
    priors = loio_bestarea_priors(metric_rows)
    valid_cache: dict[tuple[str, str], np.ndarray] = {}
    rows = [recompute_row(row, priors, valid_cache) for row in metric_rows]
    write_csv(OUT_METRICS, rows, list(METRIC_FIELDS) + ["PreviousDice_classwise_toparea", "PreviousmIoU", "V2StyleAreaFraction", "score_file_error"])
    summary = summarize(rows)
    fields = [
        "target_class",
        "class_name",
        "scale",
        "n",
        "dice_classwise_toparea",
        "previous_dice",
        "delta_dice",
        "auc_auroc",
        "map",
        "miou",
        "previous_miou",
        "delta_miou",
        "precision",
        "recall",
        "area_fraction",
        "best_dice",
    ]
    write_csv(OUT_SUMMARY, summary, fields)
    print(f"wrote {OUT_METRICS}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
