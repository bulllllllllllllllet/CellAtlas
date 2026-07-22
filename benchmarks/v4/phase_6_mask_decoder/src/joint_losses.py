"""Pixel-supervised objectives for differentiable Phase-2→6 training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmarks.v4.phase_5_prompt_encoder.src.losses import prompt_region_loss


def balanced_pixel_bce(logits: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    positive = truth & valid; negative = ~truth & valid
    if not positive.flatten(1).any(1).all() or not negative.flatten(1).any(1).all():
        raise ValueError("each pixel episode must contain positive and negative valid pixels")
    positive_loss = F.softplus(-logits); negative_loss = F.softplus(logits)
    pos = (positive_loss * positive).flatten(1).sum(1) / positive.flatten(1).sum(1).clamp_min(1)
    neg = (negative_loss * negative).flatten(1).sum(1) / negative.flatten(1).sum(1).clamp_min(1)
    return (0.5 * (pos + neg)).mean()


def soft_pixel_dice(probability: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = probability * valid.to(probability.dtype)
    target = truth.to(probability.dtype) * valid.to(probability.dtype)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def soft_boundary_loss(probability: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Balanced supervision of horizontal/vertical probability changes."""
    losses = []
    for predicted_change, true_change, pair_valid in (
        (
            (probability[:, 1:] - probability[:, :-1]).abs(),
            truth[:, 1:] != truth[:, :-1],
            valid[:, 1:] & valid[:, :-1],
        ),
        (
            (probability[:, :, 1:] - probability[:, :, :-1]).abs(),
            truth[:, :, 1:] != truth[:, :, :-1],
            valid[:, :, 1:] & valid[:, :, :-1],
        ),
    ):
        edge = true_change & pair_valid; nonedge = ~true_change & pair_valid
        predicted_change = predicted_change.clamp(1e-5, 1.0 - 1e-5)
        if edge.any():
            losses.append(-predicted_change[edge].log().mean())
        if nonedge.any():
            losses.append(-(1.0 - predicted_change[nonedge]).log().mean())
    return torch.stack(losses).mean()


def assignment_regularizers(assignment: torch.Tensor) -> dict[str, torch.Tensor]:
    if assignment.ndim != 4:
        raise ValueError("assignment must be [B,K,H,W]")
    batch, regions, height, width = assignment.shape
    mass = assignment.sum((2, 3)).clamp_min(1e-6)
    fraction = mass / mass.sum(1, keepdim=True)
    balance = (fraction * (fraction * regions).clamp_min(1e-8).log()).sum(1).mean()
    entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(1).mean()
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=assignment.device, dtype=assignment.dtype),
        torch.linspace(-1, 1, width, device=assignment.device, dtype=assignment.dtype), indexing="ij",
    )
    cx = (assignment * xx).sum((2, 3)) / mass; cy = (assignment * yy).sum((2, 3)) / mass
    distance = (xx - cx[:, :, None, None]).square() + (yy - cy[:, :, None, None]).square()
    compactness = ((assignment * distance).sum((2, 3)) / mass).mean()
    return {"assignment_balance": balance, "assignment_entropy": entropy, "assignment_compactness": compactness}


def prompt_separation_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalize positive/negative prompt mass assigned to the same online region."""
    positive = output["online_positive_prompt_soft_weights"]
    negative = output["online_negative_prompt_soft_weights"]
    positive_count = batch["positive_mask"].sum(1, keepdim=True).clamp_min(1).to(positive.dtype)
    negative_count = batch["negative_mask"].sum(1, keepdim=True).clamp_min(1).to(negative.dtype)
    positive_coverage = positive.sum(1) / positive_count
    negative_coverage = negative.sum(1) / negative_count
    return (positive_coverage * negative_coverage).sum(1).mean()


def prompt_conflict_margin_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the easier prompt in each conflicting pair toward a Voronoi boundary."""
    region_xy = output["online_region_xy"].float()

    def nearest_gap(prompt_xy: torch.Tensor) -> torch.Tensor:
        nearest = torch.cdist(prompt_xy.float(), region_xy).topk(2, dim=-1, largest=False).values
        return nearest[..., 1] - nearest[..., 0]

    positive_gap = nearest_gap(batch["positive_xy"])
    negative_gap = nearest_gap(batch["negative_xy"])
    same_slot = (
        (output["online_positive_slot_indices"][:, :, None]
         == output["online_negative_slot_indices"][:, None, :])
        & batch["positive_mask"][:, :, None]
        & batch["negative_mask"][:, None, :]
    )
    pair_gap = torch.minimum(positive_gap[:, :, None], negative_gap[:, None, :])
    loss = pair_gap[same_slot].mean() if same_slot.any() else region_xy.sum() * 0.0
    return loss, same_slot.sum()


def prompt_safe_geometry_anchor_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve the teacher slot's Voronoi distance advantage for safe prompts."""
    required = (
        "geometry_teacher_prompt_conflicts", "geometry_teacher_region_xy",
        "geometry_teacher_positive_slot_indices", "geometry_teacher_negative_slot_indices",
    )
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError(f"prompt geometry anchor requires frozen teacher outputs: {missing}")
    safe_episodes = ~output["geometry_teacher_prompt_conflicts"].any(1)
    losses = []
    selected_prompts = 0
    online_xy = output["online_region_xy"].float()
    teacher_xy = output["geometry_teacher_region_xy"].float()
    for prefix in ("positive", "negative"):
        valid = batch[f"{prefix}_mask"] & safe_episodes[:, None]
        if not valid.any():
            continue
        prompt_xy = batch[f"{prefix}_xy"].float()
        teacher_slots = output[f"geometry_teacher_{prefix}_slot_indices"].clamp_min(0)
        online_distance = torch.cdist(prompt_xy, online_xy)
        teacher_distance = torch.cdist(prompt_xy, teacher_xy)
        target_online = online_distance.gather(-1, teacher_slots.unsqueeze(-1)).squeeze(-1)
        target_teacher = teacher_distance.gather(-1, teacher_slots.unsqueeze(-1)).squeeze(-1)
        target_mask = F.one_hot(teacher_slots, online_xy.shape[1]).bool()
        alternative_online = online_distance.masked_fill(target_mask, torch.inf).min(-1).values
        alternative_teacher = teacher_distance.masked_fill(target_mask, torch.inf).min(-1).values
        online_gap = alternative_online - target_online
        teacher_gap = (alternative_teacher - target_teacher).detach()
        losses.append(F.relu(teacher_gap - online_gap)[valid])
        selected_prompts += int(valid.sum())
    if losses:
        loss = torch.cat(losses).mean()
    else:
        loss = output["online_region_xy"].sum() * 0.0
    return loss, safe_episodes.sum(), torch.tensor(selected_prompts, device=loss.device)


def signed_prompt_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Require online positive/negative prompt slots to retain their signed response."""
    logits = output["logits"]
    positive = logits.gather(1, output["online_positive_slot_indices"].clamp_min(0))
    negative = logits.gather(1, output["online_negative_slot_indices"].clamp_min(0))
    positive_loss = F.softplus(-positive)[batch["positive_mask"]].mean()
    negative_loss = F.softplus(negative)[batch["negative_mask"]].mean()
    return 0.5 * (positive_loss + negative_loss)


def teacher_consistency_losses(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distill Phase5 only where cached and online region semantics still agree."""
    online = output["online_binary_target"]
    cached = batch["binary_target"]
    stable = (
        (online != int(ignore_index))
        & (cached != int(ignore_index))
        & (online == cached)
        & ~output["online_all_prompted_regions"]
    )
    if stable.any():
        logit = F.smooth_l1_loss(output["logits"][stable], output["teacher_logits"][stable])
    else:
        logit = output["logits"].sum() * 0.0
    task = F.smooth_l1_loss(output["task_token"], output["teacher_task_token"])
    return logit, task, stable.sum()


def joint_pixel_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    ignore_index: int,
    weights: dict[str, float],
    region_ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pixel_truth = batch["pixel_gt"] == batch["target_class"][:, None, None]
    pixel_valid = batch["pixel_gt"] != int(ignore_index)
    pixel_bce = balanced_pixel_bce(output["pixel_logits"], pixel_truth, pixel_valid)
    pixel_dice = soft_pixel_dice(output["pixel_probability"], pixel_truth, pixel_valid)
    boundary = soft_boundary_loss(output["pixel_probability"], pixel_truth, pixel_valid)
    region_target = output.get("online_binary_target", batch["binary_target"])
    prompted_regions = output.get("online_prompted_regions", batch["prompted_regions"])
    region, region_parts = prompt_region_loss(
        output, region_target, prompted_regions, int(ignore_index),
        dice_weight=1.0, ranking_weight=float(weights.get("region_ranking", 0.0)),
        ranking_margin=float(region_ranking_margin),
        allow_single_class_episodes=True,
    )
    regularizers = assignment_regularizers(output["assignment_low"])
    separation = prompt_separation_loss(output, batch)
    conflict_margin, conflict_margin_pairs = prompt_conflict_margin_loss(output, batch)
    if float(weights.get("prompt_geometry_anchor", 0.0)) > 0:
        geometry_anchor, geometry_anchor_episodes, geometry_anchor_regions = prompt_safe_geometry_anchor_loss(output, batch)
    else:
        geometry_anchor = output["online_region_xy"].sum() * 0.0
        geometry_anchor_episodes = torch.zeros((), device=geometry_anchor.device)
        geometry_anchor_regions = torch.zeros((), device=geometry_anchor.device)
    signed = signed_prompt_loss(output, batch)
    teacher_logit, teacher_task, teacher_stable = teacher_consistency_losses(output, batch, ignore_index)
    conflicts = output["online_prompt_conflicts"]
    total = (
        float(weights["pixel_bce"]) * pixel_bce
        + float(weights["pixel_dice"]) * pixel_dice
        + float(weights["pixel_boundary"]) * boundary
        + float(weights["region_aux"]) * region
        + float(weights["assignment_balance"]) * regularizers["assignment_balance"]
        + float(weights["assignment_entropy"]) * regularizers["assignment_entropy"]
        + float(weights["assignment_compactness"]) * regularizers["assignment_compactness"]
        + float(weights.get("prompt_separation", 0.0)) * separation
        + float(weights.get("prompt_conflict_margin", 0.0)) * conflict_margin
        + float(weights.get("prompt_geometry_anchor", 0.0)) * geometry_anchor
        + float(weights.get("prompt_sign", 0.0)) * signed
        + float(weights.get("teacher_logit", 0.0)) * teacher_logit
        + float(weights.get("teacher_task", 0.0)) * teacher_task
    )
    return total, {
        "pixel_bce": pixel_bce.detach(), "pixel_dice_loss": pixel_dice.detach(),
        "pixel_boundary_loss": boundary.detach(), "region_aux_loss": region.detach(),
        **{name: value.detach() for name, value in regularizers.items()},
        "prompt_separation_loss": separation.detach(),
        "prompt_conflict_margin_loss": conflict_margin.detach(),
        "prompt_conflict_margin_pairs": conflict_margin_pairs.detach(),
        "prompt_geometry_anchor_loss": geometry_anchor.detach(),
        "prompt_geometry_anchor_episodes": geometry_anchor_episodes.detach(),
        "prompt_geometry_anchor_prompts": geometry_anchor_regions.detach(),
        "prompt_sign_loss": signed.detach(),
        "teacher_logit_loss": teacher_logit.detach(),
        "teacher_task_loss": teacher_task.detach(),
        "teacher_stable_regions": teacher_stable.detach(),
        "prompt_conflict_slots": conflicts.sum().detach(),
        "prompt_conflict_episodes": conflicts.any(1).sum().detach(),
        "prompt_episodes": torch.tensor(conflicts.shape[0], device=conflicts.device),
        **{f"region_{name}": value for name, value in region_parts.items()},
    }
