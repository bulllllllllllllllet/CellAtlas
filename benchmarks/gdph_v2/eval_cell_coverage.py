from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


NUM_CLASSES = 12


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sampled_oracle_coverage(
    gt_mask: np.ndarray,
    cell_xy_gt: np.ndarray,
    cell_labels: np.ndarray,
    radii: list[float],
    stride: int = 8,
) -> list[dict]:
    sampled = np.asarray(gt_mask[::stride, ::stride])
    grid_y, grid_x = np.indices(sampled.shape)
    points = np.column_stack([grid_x.reshape(-1) * stride, grid_y.reshape(-1) * stride])
    true = sampled.reshape(-1).astype(np.int64)
    valid_pixels = (true >= 0) & (true < NUM_CLASSES)
    points = points[valid_pixels]
    true = true[valid_pixels]
    valid_cells = (cell_labels >= 0) & (cell_labels < NUM_CLASSES)
    if not np.any(valid_cells):
        raise ValueError("no valid cells for coverage evaluation")
    valid_xy = np.asarray(cell_xy_gt[valid_cells], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(valid_xy)
        distance, nearest = tree.query(points, k=1, workers=-1)
    except ModuleNotFoundError:
        if len(points) * len(valid_xy) > 1_000_000:
            raise RuntimeError("scipy is required for production-scale cell coverage evaluation")
        squared = ((points[:, None, :] - valid_xy[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(squared, axis=1)
        distance = np.sqrt(squared[np.arange(len(points)), nearest])
    nearest_label = np.asarray(cell_labels[valid_cells], dtype=np.int64)[nearest]
    gt_totals = np.bincount(true, minlength=NUM_CLASSES).astype(np.int64)
    results = []
    for radius in radii:
        covered = distance <= radius
        covered_count = int(covered.sum())
        correct = covered & (nearest_label == true)
        per_class_iou = {}
        per_class = {}
        for class_id in range(NUM_CLASSES):
            class_pixels = true == class_id
            class_total = int(class_pixels.sum())
            tp = int(np.sum(correct & class_pixels))
            pred = int(np.sum(covered & (nearest_label == class_id)))
            union = int(gt_totals[class_id]) + pred - tp
            if class_total > 0:
                per_class_iou[class_id] = tp / union if union else 0.0
                class_covered = covered & class_pixels
                class_covered_count = int(class_covered.sum())
                class_distances = distance[class_pixels]
                per_class[str(class_id)] = {
                    "sampled_pixels": class_total,
                    "cell_tokens_with_class": int(np.sum(cell_labels[valid_cells] == class_id)),
                    "covered_pixels": class_covered_count,
                    "coverage": class_covered_count / class_total,
                    "covered_oracle_accuracy": (
                        float(np.mean(nearest_label[class_covered] == class_id))
                        if class_covered_count
                        else 0.0
                    ),
                    "whole_image_oracle_accuracy": tp / class_total,
                    "oracle_iou": per_class_iou[class_id],
                    "nearest_cell_distance_gt_pixels_p50": float(np.quantile(class_distances, 0.5)),
                    "nearest_cell_distance_gt_pixels_p90": float(np.quantile(class_distances, 0.9)),
                    "nearest_cell_distance_gt_pixels_p95": float(np.quantile(class_distances, 0.95)),
                }
        results.append(
            {
                "radius_gt_pixels": radius,
                "sample_stride": stride,
                "sampled_valid_pixels": len(true),
                "covered_pixels": covered_count,
                "coverage": covered_count / len(true) if len(true) else 0.0,
                "covered_oracle_accuracy": float(np.mean(nearest_label[covered] == true[covered]))
                if covered_count
                else 0.0,
                "whole_image_oracle_accuracy": float(correct.sum() / len(true)) if len(true) else 0.0,
                "whole_image_oracle_mean_iou": float(np.mean(list(per_class_iou.values())))
                if per_class_iou
                else 0.0,
                "per_class_iou": per_class_iou,
                "per_class": per_class,
            }
        )
    return results


def evaluate_slide(image_id: str, output_root: Path, radii: list[float], stride: int) -> dict:
    slide_dir = output_root / "cells" / image_id
    cells = _read_csv(slide_dir / "cells.csv")
    labels = _read_csv(slide_dir / "tissue_labels.csv")
    if len(cells) != len(labels):
        raise RuntimeError(f"{image_id} cell/label mismatch")
    accepted = np.asarray([row["label_status"] == "valid" for row in labels])
    xy = np.asarray(
        [[float(row["x_gt"]), float(row["y_gt"])] for row in cells], dtype=np.float64
    )[accepted]
    tissue = np.asarray([int(row["gt_tissue_label"]) for row in labels], dtype=np.int64)[accepted]
    gt_mask = np.load(output_root / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r")
    results = sampled_oracle_coverage(gt_mask, xy, tissue, radii, stride)
    class_cell_counts = np.bincount(tissue, minlength=NUM_CLASSES).tolist()
    return {
        "image_id": image_id,
        "valid_cells": len(tissue),
        "class_cell_counts": class_cell_counts,
        "radii": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sparse cell coverage and oracle propagation.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--radii", nargs="+", type=float, default=[10, 25, 50, 100])
    parser.add_argument("--sample_stride", type=int, default=8)
    args = parser.parse_args()
    if args.sample_stride <= 0 or any(radius <= 0 for radius in args.radii):
        raise ValueError("sample_stride and radii must be positive")
    if args.radii != sorted(set(args.radii)):
        raise ValueError("radii must be unique and increasing")
    rows = _read_csv(args.manifest)
    reports = [
        evaluate_slide(row["image_id"], Path(args.output_root), args.radii, args.sample_stride)
        for row in rows
    ]
    output_dir = Path(args.output_root) / "dense_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "cell_coverage_per_image.json"
    temporary_per_image = per_image_path.with_suffix(per_image_path.suffix + ".tmp")
    with open(temporary_per_image, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)
    temporary_per_image.replace(per_image_path)
    summary = []
    for radius in args.radii:
        values = [
            next(item for item in report["radii"] if item["radius_gt_pixels"] == radius)
            for report in reports
        ]
        summary.append(
            {
                "radius_gt_pixels": radius,
                "images": len(values),
                "mean_coverage": float(np.mean([item["coverage"] for item in values])),
                "mean_covered_oracle_accuracy": float(
                    np.mean([item["covered_oracle_accuracy"] for item in values])
                ),
                "mean_whole_image_oracle_accuracy": float(
                    np.mean([item["whole_image_oracle_accuracy"] for item in values])
                ),
                "mean_whole_image_oracle_miou": float(
                    np.mean([item["whole_image_oracle_mean_iou"] for item in values])
                ),
            }
        )
    summary_path = output_dir / "cell_coverage_summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(temporary_summary, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    temporary_summary.replace(summary_path)

    palette_payload = json.loads(
        (Path(args.output_root) / "manifests" / "gdph_tissue_palette.json").read_text(
            encoding="utf-8"
        )
    )
    class_names = {int(item["id"]): item["name"] for item in palette_payload["classes"]}
    by_class = []
    for radius in args.radii:
        for class_id in range(NUM_CLASSES):
            class_values = []
            for report in reports:
                radius_result = next(
                    item for item in report["radii"] if item["radius_gt_pixels"] == radius
                )
                value = radius_result["per_class"].get(str(class_id))
                if value is not None:
                    class_values.append(value)
            if not class_values:
                continue
            by_class.append(
                {
                    "radius_gt_pixels": radius,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "images_with_class": len(class_values),
                    "mean_coverage": float(np.mean([item["coverage"] for item in class_values])),
                    "mean_covered_oracle_accuracy": float(
                        np.mean([item["covered_oracle_accuracy"] for item in class_values])
                    ),
                    "mean_whole_image_oracle_accuracy": float(
                        np.mean([item["whole_image_oracle_accuracy"] for item in class_values])
                    ),
                    "mean_nearest_cell_distance_p90": float(
                        np.mean(
                            [item["nearest_cell_distance_gt_pixels_p90"] for item in class_values]
                        )
                    ),
                    "acellular_focus": class_id in {3, 9, 10},
                }
            )
    by_class_path = output_dir / "cell_coverage_by_class.json"
    temporary_by_class = by_class_path.with_suffix(by_class_path.suffix + ".tmp")
    temporary_by_class.write_text(
        json.dumps(by_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_by_class.replace(by_class_path)

    all_finite = all(
        np.isfinite(value)
        for item in summary
        for value in item.values()
        if isinstance(value, (int, float))
    ) and all(
        np.isfinite(value)
        for item in by_class
        for value in item.values()
        if isinstance(value, (int, float))
    )
    monotonic = all(
        all(
            left["coverage"] <= right["coverage"] + 1e-12
            for left, right in zip(report["radii"], report["radii"][1:])
        )
        for report in reports
    )
    validation = {
        "images": len(reports),
        "radii_gt_pixels": args.radii,
        "sample_stride": args.sample_stride,
        "all_metrics_finite": bool(all_finite),
        "coverage_monotonic_with_radius": monotonic,
        "acellular_focus_classes": {"3": "necrosis", "9": "mucus", "10": "fat"},
        "acellular_class_rows": sum(item["acellular_focus"] for item in by_class),
    }
    validation["passed"] = bool(
        validation["images"] == 20
        and validation["all_metrics_finite"]
        and validation["coverage_monotonic_with_radius"]
        and validation["acellular_class_rows"] > 0
    )
    validation_path = output_dir / "cell_coverage_validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_validation.replace(validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"cell coverage validation failed: {validation}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
