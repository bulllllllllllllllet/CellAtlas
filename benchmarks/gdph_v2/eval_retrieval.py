from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalized_median_prototype(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)
    prototype = np.median(values, axis=0)
    return prototype / max(float(np.linalg.norm(prototype)), 1e-12)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (100, 1000)) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return {"positives": positives, "negatives": negatives, "valid": False}
    order = np.argsort(-scores, kind="stable")
    sorted_true = y_true[order]
    cumulative_true = np.cumsum(sorted_true)
    positive_ranks = np.flatnonzero(sorted_true) + 1
    average_precision = float(np.mean(cumulative_true[positive_ranks - 1] / positive_ranks))
    ascending = np.argsort(scores, kind="stable")
    sorted_scores = scores[ascending]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[ascending[start:end]] = ((start + 1) + end) / 2
        start = end
    rank_sum_positive = float(ranks[y_true].sum())
    auroc = (rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives)
    result = {
        "positives": positives,
        "negatives": negatives,
        "valid": True,
        "average_precision": average_precision,
        "auroc": auroc,
    }
    for k in ks:
        actual_k = min(k, len(order))
        hits = int(y_true[order[:actual_k]].sum())
        result[f"precision_at_{k}"] = hits / actual_k
        result[f"recall_at_{k}"] = hits / positives
    return result


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & y_true))
    fp = int(np.sum(predicted & ~y_true))
    fn = int(np.sum(~predicted & y_true))
    tn = int(np.sum(~predicted & ~y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "similarity_threshold": float(threshold),
        "binary_true_positive": tp,
        "binary_false_positive": fp,
        "binary_false_negative": fn,
        "binary_true_negative": tn,
        "binary_accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "binary_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "predicted_positive_fraction": float(np.mean(predicted)) if len(predicted) else 0.0,
    }


def load_retrieval_slide(output_root: Path, image_id: str, head: str) -> dict[str, np.ndarray]:
    slide_dir = output_root / "cells" / image_id
    cells = _read_csv(slide_dir / "cells.csv")
    labels = _read_csv(slide_dir / "tissue_labels.csv")
    features = np.load(slide_dir / f"{head}.npy", mmap_mode="r")
    if not (len(cells) == len(labels) == len(features)):
        raise RuntimeError(f"{image_id} retrieval cardinality mismatch")
    expected_indices = list(range(len(cells)))
    if [int(row["cell_index"]) for row in cells] != expected_indices or [
        int(row["cell_index"]) for row in labels
    ] != expected_indices:
        raise RuntimeError(f"{image_id} retrieval indices are not aligned")
    x = np.asarray([float(row["x_original"]) for row in cells])
    y = np.asarray([float(row["y_original"]) for row in cells])
    tissue = np.asarray([int(row["gt_tissue_label"]) for row in labels])
    valid = np.asarray([row["label_status"] == "valid" for row in labels])
    normalized = np.asarray(features[valid], dtype=np.float32)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    if len(normalized) == 0 or not np.isfinite(normalized).all():
        raise RuntimeError(f"{image_id}/{head} has no finite valid retrieval features")
    return {
        "x": x[valid],
        "y": y[valid],
        "tissue": tissue[valid],
        "features": normalized,
    }


def evaluate_query(
    query: dict[str, str], slide_data: dict[str, np.ndarray], head: str, buffer_pixels: int,
    query_similarity_quantile: float,
) -> dict:
    image_id = query["image_id"]
    x = slide_data["x"]
    y = slide_data["y"]
    tissue = slide_data["tissue"]
    features = slide_data["features"]
    x0, y0, x1, y1 = (float(query[key]) for key in ("x0_original", "y0_original", "x1_original", "y1_original"))
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    if int(inside.sum()) == 0:
        raise RuntimeError(f"{query['query_id']} has no valid query cells")
    prototype = normalized_median_prototype(features[inside])
    query_scores = features[inside] @ prototype
    threshold = float(np.quantile(query_scores, query_similarity_quantile))
    candidate = ~(
        (x >= x0 - buffer_pixels)
        & (x < x1 + buffer_pixels)
        & (y >= y0 - buffer_pixels)
        & (y < y1 + buffer_pixels)
    )
    scores = features[candidate] @ prototype
    metrics = ranking_metrics(tissue[candidate] == int(query["class_id"]), scores)
    threshold_metrics = binary_metrics(
        tissue[candidate] == int(query["class_id"]), scores, threshold
    )
    return {
        "query_id": query["query_id"],
        "image_id": image_id,
        "class_id": int(query["class_id"]),
        "box_size_original": int(query["box_size_original"]),
        "query_cells": int(inside.sum()),
        "candidate_cells": int(candidate.sum()),
        "buffer_pixels": buffer_pixels,
        "query_similarity_quantile": query_similarity_quantile,
        "head": head,
        **metrics,
        **threshold_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query-by-region GDPH cell retrieval.")
    parser.add_argument(
        "--queries_csv", default=str(DEFAULT_OUTPUT_ROOT / "region_retrieval" / "queries.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--heads", nargs="+", choices=["raw", "reg", "proj"], default=["raw", "reg", "proj"])
    parser.add_argument("--buffer_pixels", type=int, default=256)
    parser.add_argument("--query_similarity_quantile", type=float, default=0.1)
    args = parser.parse_args()
    if args.buffer_pixels < 0:
        raise ValueError("buffer_pixels must be non-negative")
    if not 0 <= args.query_similarity_quantile <= 1:
        raise ValueError("query_similarity_quantile must be in [0, 1]")
    queries = _read_csv(args.queries_csv)
    if not queries:
        raise RuntimeError("queries CSV is empty")
    cell_queries = [
        query
        for query in queries
        if query.get("cell_query_eligible", "True").lower() in {"true", "1", "yes"}
    ]
    if not cell_queries:
        raise RuntimeError("no cell-eligible queries")
    output_root = Path(args.output_root)
    results = []
    invalid_results = []
    queries_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for query in cell_queries:
        queries_by_image[query["image_id"]].append(query)
    for head in args.heads:
        for image_id, image_queries in queries_by_image.items():
            slide_data = load_retrieval_slide(output_root, image_id, head)
            for query in image_queries:
                result = evaluate_query(
                    query,
                    slide_data,
                    head,
                    args.buffer_pixels,
                    args.query_similarity_quantile,
                )
                if result["valid"]:
                    results.append(result)
                else:
                    invalid_results.append(
                        {"head": head, "query_id": query["query_id"], **result}
                    )
    if not results:
        raise RuntimeError("retrieval evaluation produced no valid query results")
    output_path = output_root / "region_retrieval" / "cell_retrieval_metrics.csv"
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    temporary_output.replace(output_path)
    grouped: dict[str, dict] = {}
    for head in args.heads:
        subset = [row for row in results if row["head"] == head]
        grouped[head] = {
            "queries": len(subset),
            "mean_average_precision": float(np.mean([row["average_precision"] for row in subset])),
            "mean_auroc": float(np.mean([row["auroc"] for row in subset])),
            "mean_precision_at_100": float(np.mean([row["precision_at_100"] for row in subset])),
            "mean_recall_at_1000": float(np.mean([row["recall_at_1000"] for row in subset])),
            "mean_binary_accuracy": float(np.mean([row["binary_accuracy"] for row in subset])),
            "mean_binary_f1": float(np.mean([row["binary_f1"] for row in subset])),
            "mean_binary_iou": float(np.mean([row["binary_iou"] for row in subset])),
        }
    summary_path = output_root / "region_retrieval" / "cell_retrieval_summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(temporary_summary, "w", encoding="utf-8") as file:
        json.dump(grouped, file, indent=2, ensure_ascii=False)
    temporary_summary.replace(summary_path)
    by_class = []
    for head in args.heads:
        for class_id in sorted({int(row["class_id"]) for row in results if row["head"] == head}):
            subset = [
                row
                for row in results
                if row["head"] == head and int(row["class_id"]) == class_id
            ]
            by_class.append(
                {
                    "head": head,
                    "class_id": class_id,
                    "queries": len(subset),
                    "mean_average_precision": float(
                        np.mean([row["average_precision"] for row in subset])
                    ),
                    "mean_binary_f1": float(np.mean([row["binary_f1"] for row in subset])),
                    "mean_binary_iou": float(np.mean([row["binary_iou"] for row in subset])),
                    "acellular_focus": class_id in {3, 9, 10},
                }
            )
    by_class_path = output_root / "region_retrieval" / "cell_retrieval_by_class.json"
    temporary_by_class = by_class_path.with_suffix(by_class_path.suffix + ".tmp")
    temporary_by_class.write_text(
        json.dumps(by_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_by_class.replace(by_class_path)

    result_keys = [(row["head"], row["query_id"]) for row in results]
    query_sets = {
        head: {row["query_id"] for row in results if row["head"] == head}
        for head in args.heads
    }
    all_finite = all(
        np.isfinite(value)
        for row in results
        for value in row.values()
        if isinstance(value, (int, float))
    )
    validation = {
        "input_queries": len(queries),
        "cell_eligible_queries": len(cell_queries),
        "cell_ineligible_queries": len(queries) - len(cell_queries),
        "heads": args.heads,
        "valid_results": len(results),
        "invalid_results": len(invalid_results),
        "unique_head_query_pairs": len(set(result_keys)),
        "same_valid_query_set_for_all_heads": len(
            {frozenset(values) for values in query_sets.values()}
        )
        == 1,
        "candidate_scope": "same slide, query box plus buffer excluded",
        "threshold_protocol": (
            f"query-region similarity quantile {args.query_similarity_quantile}; "
            "candidate GT is not used to choose threshold"
        ),
        "feature_loading": "once per slide and head",
        "all_metrics_finite": bool(all_finite),
        "class_summary_rows": len(by_class),
    }
    validation["passed"] = bool(
        validation["valid_results"] == validation["unique_head_query_pairs"] > 0
        and all(query_sets.values())
        and validation["same_valid_query_set_for_all_heads"]
        and validation["all_metrics_finite"]
        and validation["class_summary_rows"] > 0
    )
    validation_path = output_root / "region_retrieval" / "cell_retrieval_validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_validation.replace(validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"cell retrieval validation failed: {validation}")
    print(json.dumps(grouped, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
