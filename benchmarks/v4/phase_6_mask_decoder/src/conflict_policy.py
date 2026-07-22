"""Explicit training, stress-set, and inference policy for prompt conflicts."""
from __future__ import annotations

from typing import Any

import torch


ABSTAIN_REASON = "positive_negative_prompt_region_conflict"
ABSTAIN_MESSAGE = (
    "Positive and negative prompts map to the same region. "
    "Adjust or separate the prompts and run inference again."
)
STRESS_COLUMNS = (
    "epoch", "rank", "episode_index", "patch_id", "wsi_id", "target_class",
    "prompt_size", "positive_slot_indices", "negative_slot_indices", "positive_xy",
    "negative_xy", "conflict_slot_indices",
)


def select_episodes(values: dict[str, Any], selected: torch.Tensor) -> dict[str, Any]:
    """Select the batch dimension while preserving non-batched metadata."""
    if selected.ndim != 1 or selected.dtype != torch.bool:
        raise ValueError("selected must be a one-dimensional boolean tensor")
    batch_size = selected.numel(); keep = selected.detach().cpu().tolist()
    result = {}
    for name, value in values.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            result[name] = value[selected]
        elif isinstance(value, list) and len(value) == batch_size:
            result[name] = [item for item, retain in zip(value, keep, strict=True) if retain]
        else:
            result[name] = value
    return result


def conflict_free_training_batch(
    output: dict[str, Any], batch: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Return the explicit conflict-free subset used by the filtered control."""
    conflict = output["online_prompt_conflicts"]
    if conflict.ndim != 2:
        raise ValueError("online_prompt_conflicts must be [B,K]")
    keep = ~conflict.any(1)
    return keep, select_episodes(output, keep), select_episodes(batch, keep)


def conflict_stress_rows(
    output: dict[str, torch.Tensor], batch: dict[str, Any], epoch: int, rank: int
) -> list[dict[str, Any]]:
    """Serialize every validation conflict occurrence without deduplication."""
    conflict = output["online_prompt_conflicts"]
    rows = []
    for position in conflict.any(1).nonzero(as_tuple=False).flatten().tolist():
        positive_valid = batch["positive_mask"][position]
        negative_valid = batch["negative_mask"][position]
        rows.append({
            "epoch": int(epoch), "rank": int(rank),
            "episode_index": int(batch["episode_index"][position]),
            "patch_id": str(batch["patch_id"][position]),
            "wsi_id": str(batch["wsi_id"][position]),
            "target_class": int(batch["target_class"][position]),
            "prompt_size": str(batch["prompt_size"][position]),
            "positive_slot_indices": output["online_positive_slot_indices"][position][positive_valid].detach().cpu().tolist(),
            "negative_slot_indices": output["online_negative_slot_indices"][position][negative_valid].detach().cpu().tolist(),
            "positive_xy": batch["positive_xy"][position][positive_valid].detach().cpu().tolist(),
            "negative_xy": batch["negative_xy"][position][negative_valid].detach().cpu().tolist(),
            "conflict_slot_indices": conflict[position].nonzero(as_tuple=False).flatten().detach().cpu().tolist(),
        })
    return rows


def build_inference_response(
    output: dict[str, torch.Tensor], episode_index: int = 0
) -> dict[str, Any]:
    """Return a mask only when positive/negative prompts do not conflict."""
    conflict = output["online_prompt_conflicts"]
    if not 0 <= int(episode_index) < conflict.shape[0]:
        raise IndexError(f"episode_index {episode_index} outside batch size {conflict.shape[0]}")
    conflict_slots = conflict[episode_index].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    if conflict_slots:
        return {
            "status": "abstain", "abstain": True, "reason": ABSTAIN_REASON,
            "message": ABSTAIN_MESSAGE, "conflict_slot_indices": conflict_slots,
            "pixel_probability": None,
        }
    return {
        "status": "ok", "abstain": False, "reason": None, "message": None,
        "conflict_slot_indices": [],
        "pixel_probability": output["pixel_probability"][episode_index],
    }
