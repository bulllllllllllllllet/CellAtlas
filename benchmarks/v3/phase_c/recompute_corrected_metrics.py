# 作用：用修复后的 hard-majority 面积先验从已有 Phase C score npz 重算 Dice/mIoU。

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
from benchmarks.v3.phase_c.run_multiscale_baseline import (
    MANIFEST_PATH,
    METRIC_FIELDS,
    METRICS_PATH,
    MULTISCALE_ROOT,
    NUM_CLASSES,
    SCORE_DIR,
    class_area_priors,
    read_csv,
)

OUT_METRICS = Path(__file__).resolve().parent / "corrected_multiscale_baseline_metrics.csv"
OUT_SUMMARY = Path(__file__).resolve().parent / "corrected_class_scale_metric_summary.csv"
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


def scale_valid_mask(image_id: str, scale: str) -> np.ndarray:
    rows = read_csv(MULTISCALE_ROOT / image_id / scale / "superpixels.csv")
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    valid_fraction = np.asarray([float(row.get("valid_fraction", 0.0)) for row in rows], dtype=np.float32)
    return (labels >= 0) & (labels < NUM_CLASSES) & (valid_fraction > 0)


def corrected_row(row: dict[str, str], valid_cache: dict[tuple[str, str], np.ndarray], priors: dict[tuple[str, int, str], tuple[float, float]]) -> dict[str, object]:
    image_id = row["image_id"]
    scale = row["scale"]
    target_class = int(row["target_class"])
    out = dict(row)
    out["PreviousDice_classwise_toparea"] = row.get("Dice_classwise_toparea", "")
    out["PreviousmIoU"] = row.get("mIoU", "")
    if row.get("status") != "ok":
        out["CorrectedAreaFraction"] = ""
        return out

    score_path = SCORE_DIR / f"{row['query_id']}_{scale}.npz"
    try:
        score_npz = np.load(score_path)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        out["status"] = "score_file_error"
        out["CorrectedAreaFraction"] = ""
        out["score_file_error"] = str(exc)
        return out
    valid = valid_cache.setdefault((image_id, scale), scale_valid_mask(image_id, scale))
    scores = score_npz["score_final"].astype(np.float32)[valid]
    areas = score_npz["area"].astype(np.float64)[valid]
    target = score_npz["gt_hard_label_per_superpixel"].astype(bool)[valid]
    global_prior, classwise_prior = priors[(image_id, target_class, scale)]
    global_top = predict_top_area(scores, areas, global_prior)
    classwise_top = predict_top_area(scores, areas, classwise_prior)
    count_metrics = binary_segmentation_metrics(target, classwise_top)

    target_bool = np.asarray(target, dtype=bool)
    pred_bool = np.asarray(classwise_top, dtype=bool)
    pred_area = float(np.sum(areas[pred_bool]))
    gt_area = float(np.sum(areas[target_bool]))
    fp_area = float(np.sum(areas[pred_bool & ~target_bool]))
    fn_area = float(np.sum(areas[~pred_bool & target_bool]))

    out["Dice_global_toparea"] = float(binary_segmentation_metrics(target, global_top)["dice"])
    out["Dice_classwise_toparea"] = float(count_metrics["dice"])
    out["mIoU"] = float(count_metrics["iou"])
    out["PredArea"] = pred_area
    out["GTArea"] = gt_area
    out["Precision"] = float(count_metrics["precision"])
    out["Recall"] = float(count_metrics["recall"])
    out["FP_area"] = fp_area
    out["FN_area"] = fn_area
    out["gt_positive_segments"] = int(np.sum(target_bool))
    out["gt_negative_segments"] = int(np.sum(~target_bool))
    out["CorrectedAreaFraction"] = classwise_prior
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

    summary: list[dict[str, object]] = []
    for class_id in range(NUM_CLASSES):
        for scale in SCALES:
            group = groups.get((class_id, scale), [])
            if not group:
                continue
            mean = lambda key: float(np.mean([float(row[key]) for row in group]))
            summary.append(
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
                    "area_fraction": f"{mean('CorrectedAreaFraction'):.6f}",
                    "best_dice": f"{mean('BestDice'):.6f}",
                }
            )
    return summary


def main() -> None:
    manifest_rows = read_csv(MANIFEST_PATH)
    priors = class_area_priors(manifest_rows)
    metric_rows = read_csv(METRICS_PATH)
    valid_cache: dict[tuple[str, str], np.ndarray] = {}
    corrected = [corrected_row(row, valid_cache, priors) for row in metric_rows]
    metric_fields = list(METRIC_FIELDS) + ["PreviousDice_classwise_toparea", "PreviousmIoU", "CorrectedAreaFraction", "score_file_error"]
    write_csv(OUT_METRICS, corrected, metric_fields)
    summary = summarize(corrected)
    write_csv(
        OUT_SUMMARY,
        summary,
        [
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
        ],
    )
    print(f"wrote {OUT_METRICS}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
