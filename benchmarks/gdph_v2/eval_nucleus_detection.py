from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


Image.MAX_IMAGE_PIXELS = None


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def stream_instance_centroids(path: str | Path, tile_size: int = 2048) -> np.ndarray:
    import tifffile
    import zarr

    totals: dict[int, list[float]] = {}
    with tifffile.TiffFile(path) as tif:
        store = tif.series[0].aszarr()
        image = zarr.open(store, mode="r")
        if image.ndim != 2:
            store.close()
            raise ValueError(f"expected 2D instance TIFF, got shape {image.shape}")
        height, width = image.shape
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                tile = np.asarray(
                    image[y0 : min(y0 + tile_size, height), x0 : min(x0 + tile_size, width)],
                    dtype=np.int64,
                )
                ys, xs = np.nonzero(tile)
                if ys.size == 0:
                    continue
                labels = tile[ys, xs]
                unique, inverse = np.unique(labels, return_inverse=True)
                counts = np.bincount(inverse)
                sum_x = np.bincount(inverse, weights=xs + x0)
                sum_y = np.bincount(inverse, weights=ys + y0)
                for index, label in enumerate(unique):
                    value = totals.setdefault(int(label), [0.0, 0.0, 0.0])
                    value[0] += float(counts[index])
                    value[1] += float(sum_x[index])
                    value[2] += float(sum_y[index])
        store.close()
    output = np.empty((len(totals), 3), dtype=np.float64)
    for index, label in enumerate(sorted(totals)):
        count, sum_x, sum_y = totals[label]
        output[index] = [label, sum_x / count, sum_y / count]
    return output


def one_to_one_point_metrics(
    predicted_xy: np.ndarray, gt_xy: np.ndarray, max_distance: float
) -> dict[str, float | int]:
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    candidate_pairs: list[tuple[float, int, int]] = []
    scipy_available = True
    try:
        from scipy.spatial import cKDTree

        if len(predicted_xy) and len(gt_xy):
            neighbors = cKDTree(gt_xy).query_ball_point(
                predicted_xy, r=max_distance, workers=-1
            )
            for pred_index, gt_indices in enumerate(neighbors):
                for gt_index in gt_indices:
                    distance = float(
                        np.linalg.norm(predicted_xy[pred_index] - gt_xy[gt_index])
                    )
                    candidate_pairs.append((distance, pred_index, int(gt_index)))
    except ModuleNotFoundError:
        scipy_available = False
        # Dependency-free spatial hash fallback without allocating an NxM
        # distance matrix for whole-slide point sets.
        buckets: dict[tuple[int, int], list[int]] = {}
        for gt_index, (x, y) in enumerate(gt_xy):
            key = (int(np.floor(x / max_distance)), int(np.floor(y / max_distance)))
            buckets.setdefault(key, []).append(gt_index)
        for pred_index, (x, y) in enumerate(predicted_xy):
            bx, by = int(np.floor(x / max_distance)), int(np.floor(y / max_distance))
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    for gt_index in buckets.get((bx + offset_x, by + offset_y), []):
                        distance = float(np.linalg.norm(predicted_xy[pred_index] - gt_xy[gt_index]))
                        if distance <= max_distance:
                            candidate_pairs.append((distance, pred_index, gt_index))

    matched_distances: list[float] = []
    if candidate_pairs and scipy_available:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching

        rows = np.fromiter((item[1] for item in candidate_pairs), dtype=np.int64)
        columns = np.fromiter((item[2] for item in candidate_pairs), dtype=np.int64)
        graph = csr_matrix(
            (np.ones(len(candidate_pairs), dtype=np.uint8), (rows, columns)),
            shape=(len(predicted_xy), len(gt_xy)),
        )
        matched_gt_by_pred = maximum_bipartite_matching(graph, perm_type="column")
        matched_distances = [
            float(np.linalg.norm(predicted_xy[pred_index] - gt_xy[gt_index]))
            for pred_index, gt_index in enumerate(matched_gt_by_pred)
            if gt_index >= 0
        ]
    else:
        # Portable fallback. The production aligner environment has SciPy and
        # therefore uses maximum-cardinality matching above.
        used_pred: set[int] = set()
        used_gt: set[int] = set()
        for distance_value, pred_index, gt_index in sorted(candidate_pairs):
            if pred_index in used_pred or gt_index in used_gt:
                continue
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            matched_distances.append(distance_value)
    tp = len(matched_distances)
    fp = len(predicted_xy) - tp
    fn = len(gt_xy) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "distance_threshold": max_distance,
        "predicted": len(predicted_xy),
        "ground_truth": len(gt_xy),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_distance": float(np.mean(matched_distances)) if matched_distances else 0.0,
    }


def evaluate_slide(row: dict[str, str], output_root: Path, thresholds: list[float]) -> dict:
    centroid_cache = output_root / "masks" / f"{row['image_id']}_nucleus_gt_centroids.npy"
    if centroid_cache.exists():
        gt_centroids = np.load(centroid_cache)
    else:
        gt_centroids = stream_instance_centroids(row["nucleus_instance_path"])
        temporary = centroid_cache.with_suffix(centroid_cache.suffix + ".tmp")
        with open(temporary, "wb") as file:
            np.save(file, gt_centroids)
        temporary.replace(centroid_cache)
    if (
        gt_centroids.ndim != 2
        or gt_centroids.shape[1] != 3
        or len(gt_centroids) == 0
        or not np.isfinite(gt_centroids).all()
        or np.any(gt_centroids[:, 0] <= 0)
        or len(np.unique(gt_centroids[:, 0])) != len(gt_centroids)
    ):
        raise RuntimeError(f"invalid nucleus centroid cache for {row['image_id']}")
    gt_xy = gt_centroids[:, 1:3]

    fullres_cells = _read_csv(output_root / "cells" / row["image_id"] / "cells.csv")
    fullres_xy = np.asarray(
        [[float(cell["x_original"]), float(cell["y_original"])] for cell in fullres_cells]
    )
    old_path = Path(
        "/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/prepared/cell_labels"
    ) / f"{row['image_id']}_cell_labels.csv"
    old_cells = _read_csv(old_path)
    with Image.open(row["he_path"]) as he_image, Image.open(row["tissue_gt_path"]) as gt_image:
        original_size = he_image.size
        x_inverse = he_image.width / gt_image.width
        y_inverse = he_image.height / gt_image.height
    old_xy = np.asarray(
        [
            [float(cell["centroid_x"]) * x_inverse, float(cell["centroid_y"]) * y_inverse]
            for cell in old_cells
        ]
    )
    return {
        "image_id": row["image_id"],
        "coordinate_system": "original_level_0_pixels",
        "original_size_wh": [int(original_size[0]), int(original_size[1])],
        "ground_truth_instances": int(len(gt_xy)),
        "resized_10x_to_original_scale_xy": [x_inverse, y_inverse],
        "matching": "maximum-cardinality one-to-one bipartite matching within threshold",
        "fullres": [one_to_one_point_metrics(fullres_xy, gt_xy, value) for value in thresholds],
        "resized_10x": [one_to_one_point_metrics(old_xy, gt_xy, value) for value in thresholds],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original and resized Cellpose nucleus detection.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8, 12, 20])
    args = parser.parse_args()
    reports = [
        evaluate_slide(row, Path(args.output_root), args.thresholds)
        for row in _read_csv(args.manifest)
    ]
    output_dir = Path(args.output_root) / "cell_classification" / "scale_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_path = output_dir / "nucleus_detection.json"
    temporary_reports = reports_path.with_suffix(reports_path.suffix + ".tmp")
    with open(temporary_reports, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)
    temporary_reports.replace(reports_path)
    summary = {}
    for threshold in args.thresholds:
        summary[str(threshold)] = {}
        for method in ("fullres", "resized_10x"):
            values = [
                next(item for item in report[method] if item["distance_threshold"] == threshold)
                for report in reports
            ]
            summary[str(threshold)][method] = {
                "mean_precision": float(np.mean([item["precision"] for item in values])),
                "mean_recall": float(np.mean([item["recall"] for item in values])),
                "mean_f1": float(np.mean([item["f1"] for item in values])),
            }
            tp = sum(int(item["true_positive"]) for item in values)
            fp = sum(int(item["false_positive"]) for item in values)
            fn = sum(int(item["false_negative"]) for item in values)
            micro_precision = tp / (tp + fp) if tp + fp else 0.0
            micro_recall = tp / (tp + fn) if tp + fn else 0.0
            summary[str(threshold)][method].update(
                {
                    "pooled_true_positive": tp,
                    "pooled_false_positive": fp,
                    "pooled_false_negative": fn,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                    "micro_f1": (
                        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                        if micro_precision + micro_recall
                        else 0.0
                    ),
                }
            )
        summary[str(threshold)]["paired_delta_mean_f1_fullres_minus_resized"] = float(
            np.mean(
                [
                    next(item for item in report["fullres"] if item["distance_threshold"] == threshold)["f1"]
                    - next(item for item in report["resized_10x"] if item["distance_threshold"] == threshold)["f1"]
                    for report in reports
                ]
            )
        )
    summary_path = output_dir / "summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(temporary_summary, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    temporary_summary.replace(summary_path)
    validation = {
        "slides": len(reports),
        "thresholds_original_pixels": args.thresholds,
        "all_have_ground_truth": all(
            report["fullres"][0]["ground_truth"] > 0 for report in reports
        ),
        "all_metrics_finite": all(
            np.isfinite(value)
            for by_threshold in summary.values()
            for method_values in by_threshold.values()
            if isinstance(method_values, dict)
            for value in method_values.values()
            if isinstance(value, (int, float))
        ),
    }
    validation["passed"] = bool(
        validation["slides"] == 5
        and validation["all_have_ground_truth"]
        and validation["all_metrics_finite"]
    )
    validation_path = output_dir / "validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_validation.replace(validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"nucleus detection validation failed: {validation}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
