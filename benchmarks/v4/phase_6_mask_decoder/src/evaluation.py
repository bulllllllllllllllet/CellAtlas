"""Auditable region-, query-, and pixel-level binary segmentation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def binary_counts(
    prediction: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    if prediction.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError("prediction, truth, and valid masks must have identical shapes")
    dims = tuple(range(1, prediction.ndim))
    prediction = prediction.bool(); truth = truth.bool(); valid = valid.bool()
    return {
        "tp": (prediction & truth & valid).sum(dims),
        "fp": (prediction & ~truth & valid).sum(dims),
        "fn": (~prediction & truth & valid).sum(dims),
        "tn": (~prediction & ~truth & valid).sum(dims),
        "positive": (truth & valid).sum(dims),
        "valid": valid.sum(dims),
    }


def dice_from_counts(tp, fp, fn) -> np.ndarray:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denominator = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)


def binary_boundary(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Two-sided 4-neighbour boundary restricted to valid pixels."""
    if mask.ndim != 3 or valid.shape != mask.shape:
        raise ValueError("mask and valid must be [B,H,W]")
    mask = mask.bool(); valid = valid.bool(); edge = torch.zeros_like(mask)
    vertical = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    horizontal = (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, 1:] |= vertical; edge[:, :-1] |= vertical
    edge[:, :, 1:] |= horizontal; edge[:, :, :-1] |= horizontal
    return edge & valid


def boundary_f1(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    tolerance: int = 2,
) -> dict[str, torch.Tensor]:
    predicted_edge = binary_boundary(prediction, valid)
    true_edge = binary_boundary(truth, valid)
    kernel = 2 * int(tolerance) + 1
    predicted_dilated = F.max_pool2d(predicted_edge[:, None].float(), kernel, stride=1, padding=tolerance)[:, 0].bool()
    true_dilated = F.max_pool2d(true_edge[:, None].float(), kernel, stride=1, padding=tolerance)[:, 0].bool()
    predicted_count = predicted_edge.sum((1, 2)); true_count = true_edge.sum((1, 2))
    matched_predicted = (predicted_edge & true_dilated).sum((1, 2))
    matched_true = (true_edge & predicted_dilated).sum((1, 2))
    precision = matched_predicted.float() / predicted_count.clamp_min(1)
    recall = matched_true.float() / true_count.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    evaluable = (predicted_count > 0) & (true_count > 0)
    f1 = torch.where(evaluable, f1, torch.full_like(f1, torch.nan))
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "predicted_boundary_pixels": predicted_count,
        "true_boundary_pixels": true_count,
        "boundary_evaluable": evaluable,
    }


def summarize_episode_rows(rows: list[dict], model_names: tuple[str, ...]) -> dict:
    summary = {"episode_count": len(rows), "models": {}}
    for model_name in model_names:
        metrics = {}
        for scope in ("region", "unprompted_region", "pixel"):
            tp = np.asarray([row[f"{model_name}_{scope}_tp"] for row in rows], dtype=np.int64)
            fp = np.asarray([row[f"{model_name}_{scope}_fp"] for row in rows], dtype=np.int64)
            fn = np.asarray([row[f"{model_name}_{scope}_fn"] for row in rows], dtype=np.int64)
            positive = np.asarray([row[f"{model_name}_{scope}_positive"] for row in rows], dtype=np.int64)
            per_episode = dice_from_counts(tp, fp, fn)
            if scope == "unprompted_region":
                keep = positive > 0
                per_episode = per_episode[keep]
                evaluable = int(keep.sum())
            else:
                evaluable = int(np.isfinite(per_episode).sum())
            micro = dice_from_counts(tp.sum(), fp.sum(), fn.sum()).item()
            metrics[scope] = {
                "micro_dice": float(micro),
                "macro_dice": float(np.nanmean(per_episode)),
                "evaluable_episodes": evaluable,
                "episodes_without_unprompted_target": int((positive == 0).sum()) if scope == "unprompted_region" else 0,
                "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
            }
        boundary = np.asarray([row[f"{model_name}_boundary_f1"] for row in rows], dtype=np.float64)
        metrics["pixel_boundary"] = {
            "macro_f1_tolerance_2px": float(np.nanmean(boundary)),
            "evaluable_episodes": int(np.isfinite(boundary).sum()),
        }
        summary["models"][model_name] = metrics
    return summary
