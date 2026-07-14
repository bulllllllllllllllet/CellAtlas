from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_aaai_eval import CLASS_NAMES
from benchmarks.gdph_v2.pret_utils import PRET_DIR, NUM_CLASSES, read_csv, write_csv_atomic, write_json_atomic


def _load_image(source_root: Path, image_id: str, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_dir = source_root / PRET_DIR / image_id
    records = read_csv(image_dir / "superpixels.csv")
    valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records], dtype=bool)
    labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    tokens = np.asarray(np.load(image_dir / f"tokens_{variant}.npy", mmap_mode="r"), dtype=np.float32)
    keep = valid & (labels >= 0) & (labels < NUM_CLASSES)
    return tokens[keep], labels[keep], keep


def _dice_iou_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for class_id in range(NUM_CLASSES):
        truth = y_true == class_id
        pred = y_pred == class_id
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                "support": int(np.sum(truth)),
                "predicted": int(np.sum(pred)),
                "dice": float(dice),
                "iou": float(iou),
            }
        )
    return rows


def _mean_present(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if int(row["support"]) > 0]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised LOIO upper bound for PRET superpixel tokens.")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.source_root)
    output_dir = Path(args.output_root) / PRET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = sorted([path.name for path in (source / PRET_DIR).iterdir() if path.is_dir() and (path / f"tokens_{args.variant}.npy").exists()])
    if args.image_id:
        requested = set(args.image_id)
        image_ids = [image_id for image_id in image_ids if image_id in requested]
    if len(image_ids) < 2:
        raise RuntimeError("need at least two images for LOIO supervised upper bound")

    data = {image_id: _load_image(source, image_id, args.variant)[:2] for image_id in image_ids}
    fold_rows = []
    class_rows = []
    all_true = []
    all_pred = []
    for fold_index, heldout in enumerate(image_ids, start=1):
        train_x = np.concatenate([data[image_id][0] for image_id in image_ids if image_id != heldout], axis=0)
        train_y = np.concatenate([data[image_id][1] for image_id in image_ids if image_id != heldout], axis=0)
        test_x, test_y = data[heldout]
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=args.workers,
            random_state=args.seed + fold_index,
        )
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        per_class = _dice_iou_per_class(test_y, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y,
            pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        fold_rows.append(
            {
                "image_id": heldout,
                "variant": args.variant,
                "test_segments": int(len(test_y)),
                "accuracy": float(accuracy_score(test_y, pred)),
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
                "mean_dice_present_classes": _mean_present(per_class, "dice"),
                "macro_miou_present_classes": _mean_present(per_class, "iou"),
            }
        )
        for row in per_class:
            class_rows.append({"image_id": heldout, "variant": args.variant, **row})
        all_true.append(test_y)
        all_pred.append(pred)
        print(f"supervised_loio {fold_index}/{len(image_ids)} image_id={heldout} segments={len(test_y)}", flush=True)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    overall_per_class = _dice_iou_per_class(y_true, y_pred)
    overall = [
        {
            "variant": args.variant,
            "folds": len(image_ids),
            "segments": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
            "mean_dice_present_classes": _mean_present(overall_per_class, "dice"),
            "macro_miou_present_classes": _mean_present(overall_per_class, "iou"),
        }
    ]
    write_csv_atomic(output_dir / "pret_supervised_upper_bound_folds.csv", fold_rows)
    write_csv_atomic(output_dir / "pret_supervised_upper_bound_by_class.csv", class_rows)
    write_csv_atomic(output_dir / "pret_supervised_upper_bound_overall.csv", overall)
    write_json_atomic(
        output_dir / "pret_supervised_upper_bound_validation.json",
        {
            "passed": True,
            "variant": args.variant,
            "images": image_ids,
            "overall": overall[0],
            "note": "This is a WSI-level LOIO supervised multi-class superpixel classifier upper bound, not prompt-based target-vs-rest retrieval.",
        },
    )
    print(json.dumps({"overall": overall[0]}, indent=2))


if __name__ == "__main__":
    main()
