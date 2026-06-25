from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.tissue_seg.metrics import confusion_matrix, summarize_confusion


NUM_CLASSES = 12


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_slide_data(
    output_root: Path,
    image_id: str,
    head: str,
    include_boundary: bool = False,
    max_cells: int | None = None,
    seed: int = 20260624,
) -> tuple[np.ndarray, np.ndarray, int]:
    slide_dir = output_root / "cells" / image_id
    features = np.load(slide_dir / f"{head}.npy", mmap_mode="r")
    labels_rows = _read_csv(slide_dir / "tissue_labels.csv")
    cells_rows = _read_csv(slide_dir / "cells.csv")
    if not (len(features) == len(labels_rows) == len(cells_rows)):
        raise RuntimeError(
            f"{image_id} strict cardinality failed: features={len(features)} "
            f"labels={len(labels_rows)} cells={len(cells_rows)}"
        )
    expected_indices = list(range(len(features)))
    label_indices = [int(row["cell_index"]) for row in labels_rows]
    cell_indices = [int(row["cell_index"]) for row in cells_rows]
    if label_indices != expected_indices or cell_indices != expected_indices:
        raise RuntimeError(f"{image_id} cell indices are not contiguous and aligned")
    accepted = {"valid", "boundary"} if include_boundary else {"valid"}
    valid = np.asarray(
        [
            row["label_status"] in accepted
            and 0 <= int(row["gt_tissue_label"]) < NUM_CLASSES
            for row in labels_rows
        ],
        dtype=bool,
    )
    all_labels = np.asarray(
        [int(row["gt_tissue_label"]) for row in labels_rows], dtype=np.uint8
    )
    accepted_indices = np.flatnonzero(valid)
    available = len(accepted_indices)
    if available == 0:
        raise RuntimeError(f"{image_id} has no accepted labeled cells")
    accepted_labels = all_labels[accepted_indices]
    if max_cells is not None and available > max_cells:
        chosen = stratified_subsample_indices(accepted_labels, max_cells, seed)
        accepted_indices = accepted_indices[chosen]
        accepted_labels = accepted_labels[chosen]
    x = np.asarray(features[accepted_indices], dtype=np.float32)
    y = accepted_labels
    if not np.isfinite(x).all():
        raise RuntimeError(f"{image_id} contains non-finite features")
    return x, y, available


def stratified_subsample_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if len(labels) <= max_samples:
        return np.arange(len(labels), dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts * (max_samples / len(labels))
    allocation = np.maximum(np.floor(exact).astype(int), 1)
    while allocation.sum() > max_samples:
        candidates = np.flatnonzero(allocation > 1)
        index = candidates[np.argmax(allocation[candidates] - exact[candidates])]
        allocation[index] -= 1
    while allocation.sum() < max_samples:
        candidates = np.flatnonzero(allocation < counts)
        index = candidates[np.argmax(exact[candidates] - allocation[candidates])]
        allocation[index] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for class_id, count in zip(classes, allocation, strict=True):
        indices = np.flatnonzero(labels == class_id)
        selected.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def stable_seed(*parts: object, base: int = 20260624) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def bootstrap_mean_ci(
    values: list[float], confidence: float = 0.95, samples: int = 5000, seed: int = 20260624
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(array.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def evaluate_head(
    rows: list[dict[str, str]],
    output_root: Path,
    head: str,
    max_iter: int,
    class_weight: str | None,
    include_boundary: bool,
    max_train_cells_per_slide: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = sorted({int(row["fold"]) for row in rows})
    pooled_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    fold_results = []
    image_results = []
    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        test_rows = [row for row in rows if int(row["fold"]) == fold]
        train_ids = {row["image_id"] for row in train_rows}
        test_ids = {row["image_id"] for row in test_rows}
        if train_ids & test_ids:
            raise RuntimeError(f"fold {fold} has image leakage: {sorted(train_ids & test_ids)}")
        train_data = [
            load_slide_data(
                output_root,
                row["image_id"],
                head,
                include_boundary,
                max_cells=max_train_cells_per_slide,
                seed=stable_seed(head, fold, row["image_id"]),
            )
            for row in train_rows
        ]
        x_train = np.concatenate([item[0] for item in train_data])
        y_train = np.concatenate([item[1] for item in train_data])
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, class_weight=class_weight, n_jobs=-1),
        )
        classifier.fit(x_train, y_train)
        fold_confusion = np.zeros_like(pooled_confusion)
        for row in test_rows:
            x_test, y_test, _ = load_slide_data(
                output_root, row["image_id"], head, include_boundary
            )
            prediction = classifier.predict(x_test)
            image_confusion = confusion_matrix(y_test, prediction, NUM_CLASSES)
            summary = summarize_confusion(image_confusion)
            image_results.append(
                {
                    "fold": fold,
                    "image_id": row["image_id"],
                    "cells": len(y_test),
                    "accuracy": summary.pixel_accuracy,
                    "mean_iou": summary.mean_iou,
                    "macro_f1": summary.mean_dice,
                }
            )
            fold_confusion += image_confusion
        pooled_confusion += fold_confusion
        fold_summary = summarize_confusion(fold_confusion)
        fold_results.append(
            {
                "fold": fold,
                "train_images": len(train_rows),
                "test_images": len(test_rows),
                "train_image_ids": sorted(train_ids),
                "test_image_ids": sorted(test_ids),
                "train_cells": int(len(y_train)),
                "train_cells_available_before_cap": int(sum(item[2] for item in train_data)),
                "max_train_cells_per_slide": max_train_cells_per_slide,
                "test_cells": int(fold_confusion.sum()),
                "train_classes": sorted(np.unique(y_train).astype(int).tolist()),
                "accuracy": fold_summary.pixel_accuracy,
                "mean_iou": fold_summary.mean_iou,
                "macro_f1": fold_summary.mean_dice,
            }
        )
    pooled = summarize_confusion(pooled_confusion)
    image_miou = bootstrap_mean_ci([row["mean_iou"] for row in image_results])
    image_f1 = bootstrap_mean_ci([row["macro_f1"] for row in image_results])
    evaluated_cells = int(pooled_confusion.sum())
    expected_cells = int(sum(row["cells"] for row in image_results))
    finite_metrics = np.asarray(
        [
            pooled.pixel_accuracy,
            pooled.mean_iou,
            pooled.mean_dice,
            *image_miou,
            *image_f1,
        ],
        dtype=np.float64,
    )
    validation = {
        "fold_count": len(folds),
        "images_evaluated": len(image_results),
        "unique_images_evaluated": len({row["image_id"] for row in image_results}),
        "evaluated_cells": evaluated_cells,
        "expected_cells": expected_cells,
        "all_folds_train_test_disjoint": all(
            not (set(item["train_image_ids"]) & set(item["test_image_ids"]))
            for item in fold_results
        ),
        "preprocessing_fit_scope": "training_fold_only_via_sklearn_pipeline",
        "metrics_finite": bool(np.isfinite(finite_metrics).all()),
    }
    validation["passed"] = bool(
        validation["fold_count"] == 5
        and validation["images_evaluated"] == validation["unique_images_evaluated"] == 20
        and validation["evaluated_cells"] == validation["expected_cells"] > 0
        and validation["all_folds_train_test_disjoint"]
        and validation["metrics_finite"]
    )
    return {
        "head": head,
        "class_weight": class_weight,
        "include_boundary": include_boundary,
        "training_sampling": "deterministic per-slide proportional stratified cap",
        "max_train_cells_per_slide": max_train_cells_per_slide,
        "pooled_accuracy": pooled.pixel_accuracy,
        "pooled_mean_iou": pooled.mean_iou,
        "pooled_macro_f1": pooled.mean_dice,
        "per_class_iou": pooled.per_class_iou,
        "per_class_f1": pooled.per_class_dice,
        "image_mean_iou": image_miou[0],
        "image_mean_iou_ci95": [image_miou[1], image_miou[2]],
        "image_macro_f1": image_f1[0],
        "image_macro_f1_ci95": [image_f1[1], image_f1[2]],
        "folds": fold_results,
        "images": image_results,
        "validation": validation,
        "confusion": pooled_confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict image-level five-fold GDPH linear probes.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--heads", nargs="+", choices=["raw", "reg", "proj"], default=["raw", "reg", "proj"])
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--include_boundary", action="store_true")
    parser.add_argument("--max_train_cells_per_slide", type=int, default=25000)
    args = parser.parse_args()
    if args.max_train_cells_per_slide <= 0:
        raise ValueError("max_train_cells_per_slide must be positive")
    rows = _read_csv(args.manifest)
    if len(rows) != 20 or len({row["fold"] for row in rows}) != 5:
        raise ValueError("main manifest must contain 20 images and five folds")
    results_root = Path(args.output_root) / "cell_classification"
    results_root.mkdir(parents=True, exist_ok=True)
    for head in args.heads:
        result = evaluate_head(
            rows,
            Path(args.output_root),
            head,
            args.max_iter,
            None if args.class_weight == "none" else args.class_weight,
            args.include_boundary,
            args.max_train_cells_per_slide,
        )
        confusion = result.pop("confusion")
        suffix = "with_boundary" if args.include_boundary else "valid_only"
        result_dir = results_root / f"{head}_{args.class_weight}_{suffix}"
        result_dir.mkdir(parents=True, exist_ok=True)
        confusion_path = result_dir / "confusion.npy"
        temporary_confusion = confusion_path.with_suffix(confusion_path.suffix + ".tmp")
        with open(temporary_confusion, "wb") as file:
            np.save(file, confusion)
        temporary_confusion.replace(confusion_path)
        metrics_path = result_dir / "metrics.json"
        temporary_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        with open(temporary_metrics, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        temporary_metrics.replace(metrics_path)
        if not result["validation"]["passed"]:
            raise RuntimeError(f"{head} cross-validation failed: {result['validation']}")
        print(json.dumps({key: value for key, value in result.items() if key not in {"folds", "images"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
