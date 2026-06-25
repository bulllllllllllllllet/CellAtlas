from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.eval_retrieval import binary_metrics, normalized_median_prototype, ranking_metrics
from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def percentile_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / max(1, len(values) - 1)


def evaluate_patch_query(
    query: dict[str, str], output_root: Path, cell_head: str, alpha: float,
    max_cell_distance: float, buffer_pixels: float, query_similarity_quantile: float,
) -> list[dict]:
    from scipy.spatial import cKDTree

    image_id = query["image_id"]
    patch_dir = output_root / "patches" / image_id
    slide_dir = output_root / "cells" / image_id
    patches = _read_csv(patch_dir / "patches.csv")
    patch_features = np.load(patch_dir / "raw.npy", mmap_mode="r")
    cells = _read_csv(slide_dir / "cells.csv")
    labels = _read_csv(slide_dir / "tissue_labels.csv")
    cell_features = np.load(slide_dir / f"{cell_head}.npy", mmap_mode="r")
    if len(patches) != len(patch_features) or not (len(cells) == len(labels) == len(cell_features)):
        raise RuntimeError(f"{image_id} patch retrieval cardinality mismatch")
    x0, y0, x1, y1 = (float(query[key]) for key in ("x0_original", "y0_original", "x1_original", "y1_original"))
    patch_xy = np.asarray(
        [[float(row["center_x_original"]), float(row["center_y_original"])] for row in patches]
    )
    patch_boxes = np.asarray(
        [
            [
                float(row["x0_original"]),
                float(row["y0_original"]),
                float(row["x1_original"]),
                float(row["y1_original"]),
            ]
            for row in patches
        ]
    )
    patch_valid = np.asarray(
        [float(row["gt_label_purity"]) >= 0.7 and 0 <= int(row["gt_tissue_label"]) < 12 for row in patches]
    )
    patch_query = patch_valid & (patch_xy[:, 0] >= x0) & (patch_xy[:, 0] < x1) & (patch_xy[:, 1] >= y0) & (patch_xy[:, 1] < y1)
    if not np.any(patch_query):
        overlap_width = np.maximum(
            0.0, np.minimum(patch_boxes[:, 2], x1) - np.maximum(patch_boxes[:, 0], x0)
        )
        overlap_height = np.maximum(
            0.0, np.minimum(patch_boxes[:, 3], y1) - np.maximum(patch_boxes[:, 1], y0)
        )
        overlap_area = overlap_width * overlap_height
        overlap_area[~patch_valid] = 0
        if overlap_area.max(initial=0) <= 0:
            return []
        patch_query[int(np.argmax(overlap_area))] = True
    patch_prototype = normalized_median_prototype(np.asarray(patch_features[patch_query]))
    normalized_patch = np.asarray(patch_features, dtype=np.float32)
    normalized_patch /= np.maximum(np.linalg.norm(normalized_patch, axis=1, keepdims=True), 1e-12)
    patch_score = normalized_patch @ patch_prototype

    cell_xy = np.asarray([[float(row["x_original"]), float(row["y_original"])] for row in cells])
    cell_valid = np.asarray([row["label_status"] == "valid" for row in labels])
    cell_query = cell_valid & (cell_xy[:, 0] >= x0) & (cell_xy[:, 0] < x1) & (cell_xy[:, 1] >= y0) & (cell_xy[:, 1] < y1)
    valid_cell_xy = cell_xy[cell_valid]
    if len(valid_cell_xy) == 0:
        raise RuntimeError(f"{image_id} has no valid cells for patch coverage")
    distance, nearest = cKDTree(valid_cell_xy).query(patch_xy, k=1, workers=-1)
    cell_available = distance <= max_cell_distance

    overlaps_buffered_query = (
        (patch_boxes[:, 2] > x0 - buffer_pixels)
        & (patch_boxes[:, 0] < x1 + buffer_pixels)
        & (patch_boxes[:, 3] > y0 - buffer_pixels)
        & (patch_boxes[:, 1] < y1 + buffer_pixels)
    )
    candidate = patch_valid & ~overlaps_buffered_query
    target = np.asarray([int(row["gt_tissue_label"]) for row in patches]) == int(query["class_id"])
    patch_rank = percentile_scores(patch_score)
    methods = [("patch_raw", patch_score, patch_score[patch_query])]
    cell_query_available = bool(np.any(cell_query))
    if cell_query_available:
        cell_prototype = normalized_median_prototype(np.asarray(cell_features[cell_query]))
        normalized_cells = np.asarray(cell_features[cell_valid], dtype=np.float32)
        normalized_cells /= np.maximum(
            np.linalg.norm(normalized_cells, axis=1, keepdims=True), 1e-12
        )
        cell_score = normalized_cells @ cell_prototype
        cell_query_valid = (
            (valid_cell_xy[:, 0] >= x0)
            & (valid_cell_xy[:, 0] < x1)
            & (valid_cell_xy[:, 1] >= y0)
            & (valid_cell_xy[:, 1] < y1)
        )
        nearest_cell_score = cell_score[nearest]
        cell_rank = percentile_scores(nearest_cell_score)
        hybrid = patch_rank.copy()
        hybrid[cell_available] = (
            alpha * patch_rank[cell_available] + (1 - alpha) * cell_rank[cell_available]
        )
        methods.extend(
            [
                (
                    f"cell_{cell_head}_nearest_uncapped",
                    nearest_cell_score,
                    cell_score[cell_query_valid],
                ),
                (f"hybrid_{cell_head}", hybrid, hybrid[patch_query]),
            ]
        )
    outputs = []
    for method, score, query_score in methods:
        metrics = ranking_metrics(target[candidate], score[candidate])
        if metrics["valid"]:
            threshold = float(np.quantile(query_score, query_similarity_quantile))
            threshold_metrics = binary_metrics(
                target[candidate], score[candidate], threshold
            )
            outputs.append(
                {
                    "query_id": query["query_id"],
                    "image_id": image_id,
                    "class_id": int(query["class_id"]),
                    "box_size_original": int(query["box_size_original"]),
                    "method": method,
                    "query_patch_tokens": int(patch_query.sum()),
                    "candidate_patches": int(candidate.sum()),
                    "buffer_pixels": buffer_pixels,
                    "max_cell_distance": max_cell_distance,
                    "query_similarity_quantile": query_similarity_quantile,
                    "cell_query_available": cell_query_available,
                    "cell_coverage_at_patches": float(np.mean(cell_available[candidate])),
                    **metrics,
                    **threshold_metrics,
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate patch and cell+patch region retrieval.")
    parser.add_argument(
        "--queries_csv", default=str(DEFAULT_OUTPUT_ROOT / "region_retrieval" / "queries.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cell_heads", nargs="+", choices=["raw", "reg", "proj"], default=["raw", "reg", "proj"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_cell_distance", type=float, default=200.0)
    parser.add_argument("--buffer_pixels", type=float, default=256.0)
    parser.add_argument("--query_similarity_quantile", type=float, default=0.1)
    args = parser.parse_args()
    if not (0 <= args.alpha <= 1):
        raise ValueError("alpha must be in [0, 1]")
    if args.max_cell_distance <= 0 or args.buffer_pixels < 0:
        raise ValueError("max_cell_distance must be positive and buffer_pixels non-negative")
    if not 0 <= args.query_similarity_quantile <= 1:
        raise ValueError("query_similarity_quantile must be in [0, 1]")
    queries = _read_csv(args.queries_csv)
    results = []
    seen_results: set[tuple[str, str]] = set()
    for head in args.cell_heads:
        for query in queries:
            query_results = evaluate_patch_query(
                query, Path(args.output_root), head, args.alpha,
                args.max_cell_distance, args.buffer_pixels, args.query_similarity_quantile,
            )
            for result in query_results:
                key = (result["method"], result["query_id"])
                if key in seen_results:
                    continue
                seen_results.add(key)
                results.append(result)
    if not results:
        raise RuntimeError("patch/hybrid retrieval produced no valid results")
    output_dir = Path(args.output_root) / "dense_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "patch_hybrid_retrieval.csv"
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with open(temporary_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    temporary_csv.replace(output_csv)
    summary = {}
    for method in sorted({row["method"] for row in results}):
        subset = [row for row in results if row["method"] == method]
        summary[method] = {
            "queries": len(subset),
            "mean_average_precision": float(np.mean([row["average_precision"] for row in subset])),
            "mean_auroc": float(np.mean([row["auroc"] for row in subset])),
            "mean_cell_coverage": float(np.mean([row["cell_coverage_at_patches"] for row in subset])),
            "mean_binary_accuracy": float(np.mean([row["binary_accuracy"] for row in subset])),
            "mean_binary_f1": float(np.mean([row["binary_f1"] for row in subset])),
            "mean_binary_iou": float(np.mean([row["binary_iou"] for row in subset])),
        }
    summary_path = output_dir / "patch_hybrid_summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(temporary_summary, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    temporary_summary.replace(summary_path)
    by_class = []
    for method in sorted({row["method"] for row in results}):
        for class_id in sorted(
            {int(row["class_id"]) for row in results if row["method"] == method}
        ):
            subset = [
                row
                for row in results
                if row["method"] == method and int(row["class_id"]) == class_id
            ]
            by_class.append(
                {
                    "method": method,
                    "class_id": class_id,
                    "queries": len(subset),
                    "mean_average_precision": float(
                        np.mean([row["average_precision"] for row in subset])
                    ),
                    "mean_binary_f1": float(np.mean([row["binary_f1"] for row in subset])),
                    "mean_binary_iou": float(np.mean([row["binary_iou"] for row in subset])),
                    "mean_cell_coverage": float(
                        np.mean([row["cell_coverage_at_patches"] for row in subset])
                    ),
                    "acellular_focus": class_id in {3, 9, 10},
                }
            )
    by_class_path = output_dir / "patch_hybrid_by_class.json"
    temporary_by_class = by_class_path.with_suffix(by_class_path.suffix + ".tmp")
    temporary_by_class.write_text(
        json.dumps(by_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_by_class.replace(by_class_path)
    expected_methods = {"patch_raw"}
    for head in args.cell_heads:
        expected_methods.update({f"cell_{head}_nearest_uncapped", f"hybrid_{head}"})
    result_pairs = [(row["method"], row["query_id"]) for row in results]
    validation = {
        "input_queries": len(queries),
        "methods": sorted(summary),
        "expected_methods": sorted(expected_methods),
        "results": len(results),
        "unique_method_query_pairs": len(set(result_pairs)),
        "alpha": args.alpha,
        "max_cell_distance_original_pixels": args.max_cell_distance,
        "query_exclusion_buffer_original_pixels": args.buffer_pixels,
        "threshold_protocol": (
            f"query-region similarity quantile {args.query_similarity_quantile}; "
            "candidate GT is not used to choose threshold"
        ),
        "query_overlap_excluded_by_patch_rectangle": True,
        "all_metrics_finite": all(
            np.isfinite(value)
            for row in results
            for value in row.values()
            if isinstance(value, (int, float))
        ),
        "acellular_focus_class_rows": sum(item["acellular_focus"] for item in by_class),
        "patch_only_zero_cell_query_results": sum(
            row["method"] == "patch_raw" and not row["cell_query_available"]
            for row in results
        ),
    }
    validation["passed"] = bool(
        set(validation["methods"]) == expected_methods
        and validation["results"] == validation["unique_method_query_pairs"] > 0
        and validation["all_metrics_finite"]
        and validation["acellular_focus_class_rows"] > 0
    )
    validation_path = output_dir / "patch_hybrid_validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_validation.replace(validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"patch/hybrid validation failed: {validation}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
