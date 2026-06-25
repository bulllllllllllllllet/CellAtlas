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


def _split_rows(rows: list[dict[str, str]], split_csv: str | None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if split_csv is None:
        if len(rows) < 2:
            raise ValueError("Need at least 2 images for default image-level split.")
        ordered = sorted(rows, key=lambda r: r["image_id"])
        cut = max(1, int(round(len(ordered) * 0.8)))
        cut = min(cut, len(ordered) - 1)
        return ordered[:cut], ordered[cut:]

    split_by_image: dict[str, str] = {}
    with open(split_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split_by_image[row["image_id"]] = row["split"]

    train = [r for r in rows if split_by_image.get(r["image_id"]) == "train"]
    test = [r for r in rows if split_by_image.get(r["image_id"]) in {"val", "test"}]
    if not train or not test:
        raise ValueError("split CSV must contain train and val/test rows.")
    return train, test


def _load_matrix(rows: list[dict[str, str]], head: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    per_image_labels: list[np.ndarray] = []
    for row in rows:
        x = np.load(_feature_path(row, head), mmap_mode="r")
        y = _load_cell_labels(row["cell_labels_path"])
        n = min(len(x), len(y))
        x = np.asarray(x[:n])
        y = y[:n]
        valid = y != IGNORE_LABEL
        features.append(x[valid])
        labels.append(y[valid])
        per_image_labels.append(y)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), per_image_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a linear probe on CellAtlas tissue labels.")
    parser.add_argument("--prepared_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head", choices=["reg", "proj", "raw"], default="reg")
    parser.add_argument("--split_csv", default=None, help="Optional CSV: image_id,split where split=train/val/test")
    parser.add_argument("--palette_json", default=None)
    parser.add_argument("--include_pixel_miou", action="store_true")
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    _require_sklearn()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = load_palette(args.palette_json)
    num_classes = len(palette)

    rows = _read_prepared_manifest(args.prepared_manifest)
    train_rows, test_rows = _split_rows(rows, args.split_csv)
    x_train, y_train, _ = _load_matrix(train_rows, args.head)
    x_test, y_test, _ = _load_matrix(test_rows, args.head)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    cell_conf = confusion_matrix(y_test, y_pred, num_classes, IGNORE_LABEL)
    cell_summary = summarize_confusion(cell_conf)
    result = {
        "head": args.head,
        "train_images": [r["image_id"] for r in train_rows],
        "test_images": [r["image_id"] for r in test_rows],
        "cell_accuracy": cell_summary.pixel_accuracy,
        "cell_mean_iou": cell_summary.mean_iou,
        "cell_macro_f1": cell_summary.mean_dice,
        "per_class_iou": {str(k): v for k, v in cell_summary.per_class_iou.items()},
        "per_class_dice": {str(k): v for k, v in cell_summary.per_class_dice.items()},
    }

    if args.include_pixel_miou:
        pixel_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for row in test_rows:
            features = np.load(_feature_path(row, args.head), mmap_mode="r")
            labels = _load_cell_labels(row["cell_labels_path"])
            n = min(len(features), len(labels))
            pred = np.full(n, IGNORE_LABEL, dtype=np.uint8)
            valid = labels[:n] != IGNORE_LABEL
            if np.any(valid):
                pred[valid] = clf.predict(np.asarray(features[:n][valid])).astype(np.uint8)

            masks = np.load(row["masks_path"], mmap_mode="r")
            gt_mask = np.load(row["gt_mask_path"], mmap_mode="r")
            pixel_conf += pixel_confusion_from_cell_masks(
                masks=masks,
                gt_mask=gt_mask,
                cell_pred_labels=pred,
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
