"""Losses and count-based metrics for Phase-5 episodic region matching."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def prompt_region_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    prompted_regions: torch.Tensor,
    ignore_index: int,
    dice_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    allow_single_class_episodes: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output["logits"]
    valid = target != ignore_index
    positive = valid & (target == 1)
    negative = valid & (target == 0)
    has_positive = positive.any(1)
    has_negative = negative.any(1)
    has_valid = valid.any(1)
    if not allow_single_class_episodes and (not has_positive.all() or not has_negative.all()):
        raise ValueError("each episode must have positive and negative target regions")
    class_bce = []
    if positive.any():
        class_bce.append(F.softplus(-logits)[positive].mean())
    if negative.any():
        class_bce.append(F.softplus(logits)[negative].mean())
    balanced_bce = torch.stack(class_bce).mean() if class_bce else logits.sum() * 0.0
    probability = logits.sigmoid() * valid.to(logits.dtype)
    truth = (target == 1).to(logits.dtype)
    intersection = (probability * truth).sum(1)
    dice_per_episode = 1.0 - ((2.0 * intersection + 1.0) / (probability.sum(1) + truth.sum(1) + 1.0))
    dice = dice_per_episode[has_valid].mean() if has_valid.any() else logits.sum() * 0.0
    similarity = output["similarity_difference"]
    positive_mean = (similarity * positive).sum(1) / positive.sum(1).clamp_min(1)
    negative_mean = (similarity * negative).sum(1) / negative.sum(1).clamp_min(1)
    ranking_evaluable = has_positive & has_negative
    ranking = (
        F.relu(float(ranking_margin) - positive_mean + negative_mean)[ranking_evaluable].mean()
        if ranking_evaluable.any() else similarity.sum() * 0.0
    )
    total = balanced_bce + float(dice_weight) * dice + float(ranking_weight) * ranking
    parts = {
        "balanced_bce": balanced_bce.detach(),
        "dice_loss": dice.detach(),
        "ranking_loss": ranking.detach(),
    }
    if allow_single_class_episodes:
        parts.update({
            "valid_episodes": has_valid.sum().detach(),
            "positive_evaluable_episodes": has_positive.sum().detach(),
            "negative_evaluable_episodes": has_negative.sum().detach(),
            "ranking_evaluable_episodes": ranking_evaluable.sum().detach(),
        })
    return total, parts


@torch.no_grad()
def metric_counts(logits: torch.Tensor, target: torch.Tensor, prompted_regions: torch.Tensor, ignore_index: int) -> dict[str, torch.Tensor]:
    valid = target != ignore_index
    truth = target == 1
    prediction = logits >= 0
    tp = (prediction & truth & valid).sum()
    fp = (prediction & ~truth & valid).sum()
    fn = (~prediction & truth & valid).sum()
    tn = (~prediction & ~truth & valid).sum()
    unprompted = truth & valid & ~prompted_regions
    unprompted_tp = (prediction & unprompted).sum()
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unprompted_tp": unprompted_tp,
        "unprompted_positive": unprompted.sum(),
    }


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp, fp, fn, tn = (float(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    return {
        "region_dice": (2 * tp) / max(2 * tp + fp + fn, 1.0),
        "region_iou": tp / max(tp + fp + fn, 1.0),
        "region_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
        "predicted_positive_fraction": (tp + fp) / max(tp + fp + fn + tn, 1.0),
        "target_recall": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
        "unprompted_target_recall": float(counts["unprompted_tp"]) / max(float(counts["unprompted_positive"]), 1.0),
    }
