"""Phase-6 graph decoder objectives."""
from __future__ import annotations

import torch

from benchmarks.v4.phase_5_prompt_encoder.src.losses import prompt_region_loss


def graph_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    adjacency: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Supervise probability changes on graph edges without indiscriminate smoothing."""
    valid = target != int(ignore_index)
    regions = logits.shape[1]
    upper = torch.triu(torch.ones(regions, regions, dtype=torch.bool, device=logits.device), diagonal=1)
    edges = adjacency & adjacency.transpose(1, 2) & upper[None]
    edges &= valid[:, :, None] & valid[:, None, :]
    if not edges.any():
        return logits.sum() * 0.0
    probability = logits.sigmoid()
    predicted_change = (probability[:, :, None] - probability[:, None, :]).abs()
    true_change = (target[:, :, None] != target[:, None, :]).to(logits.dtype)
    same = edges & (true_change == 0)
    boundary = edges & (true_change == 1)
    parts = []
    if same.any():
        parts.append(predicted_change[same].mean())
    if boundary.any():
        parts.append((1.0 - predicted_change[boundary]).mean())
    return torch.stack(parts).mean()


def decoder_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    prompted_regions: torch.Tensor,
    ignore_index: int,
    dice_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    boundary_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base, parts = prompt_region_loss(
        output, target, prompted_regions, ignore_index, dice_weight, ranking_weight, ranking_margin
    )
    boundary = graph_boundary_loss(output["logits"], target, output["adjacency"], ignore_index)
    total = base + float(boundary_weight) * boundary
    return total, {**parts, "boundary_loss": boundary.detach()}
