from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(target: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    target, pred = target.reshape(-1), pred.reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    return np.bincount(target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64), minlength=num_classes**2).reshape(num_classes, num_classes)


def summarize(conf: np.ndarray, background_ids: set[int] | None = None) -> dict[str, object]:
    background_ids = background_ids or set()
    tp, gt, pd = np.diag(conf).astype(float), conf.sum(1).astype(float), conf.sum(0).astype(float)
    denom = gt + pd
    dice = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    iou = np.divide(tp, gt + pd - tp, out=np.zeros_like(tp), where=(gt + pd - tp) > 0)
    precision = np.divide(tp, pd, out=np.zeros_like(tp), where=pd > 0)
    recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    selected = [i for i in range(len(gt)) if gt[i] > 0 and i not in background_ids]
    return {"macro_dice": float(np.mean(dice[selected])) if selected else 0., "macro_miou": float(np.mean(iou[selected])) if selected else 0., "per_class_dice": dice.tolist(), "per_class_iou": iou.tolist(), "precision": precision.tolist(), "recall": recall.tolist(), "confusion": conf.tolist()}


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int, background_ids: set[int]) -> torch.Tensor:
    probs = logits.softmax(1)
    valid = target.ne(ignore_index)
    safe = target.masked_fill(~valid, 0)
    onehot = torch.nn.functional.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probs, onehot = probs * valid, onehot * valid
    dice = (2 * (probs * onehot).sum((0, 2, 3)) + 1e-6) / (probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1e-6)
    keep = [c for c in range(num_classes) if c not in background_ids and onehot[:, c].sum() > 0]
    return 1 - dice[keep].mean() if keep else logits.sum() * 0
