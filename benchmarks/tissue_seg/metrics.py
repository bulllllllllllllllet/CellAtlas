from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    per_class_iou: dict[int, float]
    per_class_dice: dict[int, float]
    confusion: np.ndarray


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    valid = (y_true != ignore_label) & (y_true >= 0) & (y_true < num_classes)
    valid &= (y_pred >= 0) & (y_pred < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    encoded = y_true[valid].astype(np.int64) * num_classes + y_pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_confusion(confusion: np.ndarray) -> MetricSummary:
    confusion = np.asarray(confusion, dtype=np.int64)
    num_classes = confusion.shape[0]
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    union = gt + pred - tp
    dice_den = gt + pred

    per_iou: dict[int, float] = {}
    per_dice: dict[int, float] = {}
    for c in range(num_classes):
        if gt[c] > 0:
            per_iou[c] = float(tp[c] / union[c]) if union[c] > 0 else 0.0
            per_dice[c] = float(2.0 * tp[c] / dice_den[c]) if dice_den[c] > 0 else 0.0

    total = float(confusion.sum())
    pixel_accuracy = float(tp.sum() / total) if total > 0 else 0.0
    mean_iou = float(np.mean(list(per_iou.values()))) if per_iou else 0.0
    mean_dice = float(np.mean(list(per_dice.values()))) if per_dice else 0.0
    return MetricSummary(pixel_accuracy, mean_iou, mean_dice, per_iou, per_dice, confusion)


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    summary = summarize_confusion(confusion)
    return summary.mean_dice


def pixel_confusion_from_cell_masks(
    masks: np.ndarray,
    gt_mask: np.ndarray,
    cell_pred_labels: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
    row_chunk: int = 1024,
) -> np.ndarray:
    """Compute pixel confusion on cell-covered pixels without materializing a full pred image."""
    masks = np.asarray(masks)
    gt_mask = np.asarray(gt_mask)
    if masks.shape != gt_mask.shape:
        raise ValueError(f"masks shape {masks.shape} != gt_mask shape {gt_mask.shape}")

    label_to_pred = np.full(int(masks.max()) + 1, ignore_label, dtype=np.uint8)
    n = min(len(cell_pred_labels), len(label_to_pred) - 1)
    if n > 0:
        label_to_pred[1 : n + 1] = np.asarray(cell_pred_labels[:n], dtype=np.uint8)

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    h = masks.shape[0]
    for y0 in range(0, h, row_chunk):
        y1 = min(y0 + row_chunk, h)
        mask_chunk = masks[y0:y1]
        covered = mask_chunk > 0
        if not np.any(covered):
            continue
        pred_chunk = label_to_pred[mask_chunk[covered]]
        gt_chunk = gt_mask[y0:y1][covered]
        conf += confusion_matrix(gt_chunk, pred_chunk, num_classes, ignore_label)
    return conf
