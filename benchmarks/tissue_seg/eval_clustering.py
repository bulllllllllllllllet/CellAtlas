from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.tissue_seg.metrics import (
    confusion_matrix,
    pixel_confusion_from_cell_masks,
    summarize_confusion,
)
from benchmarks.tissue_seg.palette import IGNORE_LABEL, load_palette


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-learn is required. Run this in the conda env `aligner`.") from exc


def _read_prepared_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_cell_labels(path: str | Path) -> np.ndarray:
    labels: list[int] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["gt_label"]))
    return np.asarray(labels, dtype=np.uint8)


def _feature_path(row: dict[str, str], head: str) -> str:
    key = f"{head}_path"
    path = row.get(key, "")
    if not path:
        raise ValueError(f"manifest row {row.get('image_id')} has no {key}")
    return path


def _hungarian_cluster_mapping(
    y_true: np.ndarray,
    clusters: np.ndarray,
    num_classes: int,
    ignore_label: int,
) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    valid = y_true != ignore_label
    n_clusters = int(clusters.max()) + 1 if clusters.size else 0
    overlap = np.zeros((n_clusters, num_classes), dtype=np.int64)
    for cluster_id, gt_id in zip(clusters[valid], y_true[valid], strict=False):
        if 0 <= int(gt_id) < num_classes:
            overlap[int(cluster_id), int(gt_id)] += 1

    cost = overlap.max() - overlap
    cluster_idx, class_idx = linear_sum_assignment(cost)
    mapping = np.full(n_clusters, ignore_label, dtype=np.uint8)
    mapping[cluster_idx] = class_idx.astype(np.uint8)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unsupervised token clustering against tissue GT.")
    parser.add_argument("--prepared_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head", choices=["reg", "proj", "raw"], default="reg")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--palette_json", default=None)
    parser.add_argument("--include_pixel_miou", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    _require_sklearn()
    from sklearn.cluster import KMeans, MiniBatchKMeans

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = load_palette(args.palette_json)
    num_classes = len(palette)
    n_clusters = args.n_clusters or num_classes

    rows = _read_prepared_manifest(args.prepared_manifest)
    features_per_image: list[np.ndarray] = []
    labels_per_image: list[np.ndarray] = []
    image_ids: list[str] = []

    for row in rows:
        features = np.load(_feature_path(row, args.head), mmap_mode="r")
        labels = _load_cell_labels(row["cell_labels_path"])
        n = min(len(features), len(labels))
        features_per_image.append(np.asarray(features[:n]))
        labels_per_image.append(labels[:n])
        image_ids.append(row["image_id"])

    x = np.concatenate(features_per_image, axis=0)
    y = np.concatenate(labels_per_image, axis=0)
    valid = y != IGNORE_LABEL
    if not np.any(valid):
        raise SystemExit("No valid cells after GT sampling.")

    model_cls = MiniBatchKMeans if len(x) >= 100_000 else KMeans
    kwargs = {"n_clusters": n_clusters, "random_state": args.random_state}
    if model_cls is MiniBatchKMeans:
        kwargs["batch_size"] = 8192
    clusters = model_cls(**kwargs).fit_predict(x)
    mapping = _hungarian_cluster_mapping(y, clusters, num_classes, IGNORE_LABEL)
    pred = mapping[clusters]

    cell_conf = confusion_matrix(y, pred, num_classes, IGNORE_LABEL)
    cell_summary = summarize_confusion(cell_conf)

    result = {
        "head": args.head,
        "n_clusters": n_clusters,
        "cell_pixel_accuracy": cell_summary.pixel_accuracy,
        "cell_mean_iou": cell_summary.mean_iou,
        "cell_macro_f1": cell_summary.mean_dice,
        "cluster_to_class": {str(i): int(v) for i, v in enumerate(mapping.tolist())},
        "per_class_iou": {str(k): v for k, v in cell_summary.per_class_iou.items()},
        "per_class_dice": {str(k): v for k, v in cell_summary.per_class_dice.items()},
    }

    if args.include_pixel_miou:
        pixel_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        offset = 0
        for row, labels in zip(rows, labels_per_image, strict=False):
            n = len(labels)
            image_pred = pred[offset : offset + n]
            offset += n
            masks = np.load(row["masks_path"], mmap_mode="r")
            gt_mask = np.load(row["gt_mask_path"], mmap_mode="r")
            pixel_conf += pixel_confusion_from_cell_masks(
                masks=masks,
                gt_mask=gt_mask,
                cell_pred_labels=image_pred,
                num_classes=num_classes,
                ignore_label=IGNORE_LABEL,
            )
        pixel_summary = summarize_confusion(pixel_conf)
        result.update(
            {
                "pixel_accuracy": pixel_summary.pixel_accuracy,
                "pixel_mean_iou": pixel_summary.mean_iou,
                "pixel_mean_dice": pixel_summary.mean_dice,
                "pixel_per_class_iou": {str(k): v for k, v in pixel_summary.per_class_iou.items()},
                "pixel_per_class_dice": {str(k): v for k, v in pixel_summary.per_class_dice.items()},
            }
        )
        np.save(output_dir / "pixel_confusion.npy", pixel_conf)

    np.save(output_dir / "cell_confusion.npy", cell_conf)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
