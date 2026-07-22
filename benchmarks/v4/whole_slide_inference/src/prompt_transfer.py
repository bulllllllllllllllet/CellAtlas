from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import assignment_geometry
from benchmarks.v4.phase_6_mask_decoder.src.model import project_region_probabilities


@dataclass
class EncodedRegions:
    assignment: torch.Tensor
    tokens: torch.Tensor
    xy: torch.Tensor
    area: torch.Tensor
    active: torch.Tensor


def encode_regions(
    model,
    images: torch.Tensor,
    cells: torch.Tensor,
    cell_valid: torch.Tensor,
    total_cell_count: torch.Tensor,
) -> EncodedRegions:
    """Run the deployed J5 Phase2+3 path without requiring GT tensors."""
    if model.parent_context is not None:
        raise ValueError("whole-slide J5 inference requires parent_context=None")
    features = model.phase2.extract_backbone_features(images)
    phase2 = model.phase2.forward_from_features(
        features, images.shape[-2:], return_full_assignment=True, return_tokens=True
    )
    mass = sample_assignment_at_cells(phase2["assignment_low"], cells)
    fused = model.cell(
        phase2["region_tokens"], mass, cells, cell_valid, total_cell_count.float()
    )["fused_tokens"]
    xy, area = assignment_geometry(phase2["assignment_low"])
    active = area > 1e-6
    if not active.any(1).all():
        raise RuntimeError("online region encoder produced a tile without active regions")
    return EncodedRegions(phase2["assignment"], fused, xy, area, active)


def expand_prompt_task(prompt_task: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    output = {}
    for key in ("q_positive", "q_negative", "task_token"):
        value = prompt_task[key]
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"global {key} must have shape [1,D]")
        output[key] = value.expand(int(batch_size), -1)
    return output


def decode_regions_with_task(model, encoded: EncodedRegions, prompt_task: dict[str, torch.Tensor]) -> dict:
    """Apply one encoded prompt task to every candidate tile in a batch."""
    batch = {
        "fine_tokens": encoded.tokens,
        "region_xy": encoded.xy,
        "region_area": encoded.area,
        "fine_active": encoded.active,
    }
    task = expand_prompt_task(prompt_task, encoded.tokens.shape[0])
    matched = model.decoder.prompt_model.match_regions(batch, task)
    decoded = model.decoder.refine(batch, matched)
    return {**decoded, **project_region_probabilities(encoded.assignment, decoded["logits"])}
