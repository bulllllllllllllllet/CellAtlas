"""Differentiable online Phase-2→3→5→6 prompt mask path."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder, spatial_region_assignment
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells
from benchmarks.v4.phase_4_cross_scale.src.model import gather_parents
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder, project_region_probabilities


def assignment_geometry(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable normalized ``xy`` centroids and area fractions for [B,K,H,W]."""
    if assignment.ndim != 4:
        raise ValueError("assignment must be [B,K,H,W]")
    _, _, height, width = assignment.shape
    dtype = assignment.dtype; device = assignment.device
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=dtype) + 0.5) / height,
        (torch.arange(width, device=device, dtype=dtype) + 0.5) / width,
        indexing="ij",
    )
    mass = assignment.sum((2, 3)).clamp_min(1e-6)
    centroid_x = (assignment * xx).sum((2, 3)) / mass
    centroid_y = (assignment * yy).sum((2, 3)) / mass
    area = mass / mass.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.stack([centroid_x, centroid_y], dim=-1), area


def gather_prompt_tokens(
    tokens: torch.Tensor, slot_indices: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if slot_indices.shape != valid.shape or tokens.shape[0] != slot_indices.shape[0]:
        raise ValueError("prompt slot indices/mask shape mismatch")
    safe = slot_indices.clamp_min(0)
    gathered = tokens.gather(1, safe.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


def remap_prompt_tokens(
    tokens: torch.Tensor,
    region_xy: torch.Tensor,
    prompt_xy: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map cached region-centroid prompts onto current online region centroids.

    Eligibility artifacts store one centroid per prompted region rather than an
    interior pixel click or dense source-region mask. Therefore direct
    assignment sampling at that coordinate is not valid for non-convex regions.
    Forward uses the nearest online centroid; straight-through soft weights keep
    the remapping differentiable with respect to online geometry.
    """
    if tokens.ndim != 3 or region_xy.ndim != 3 or prompt_xy.ndim != 3:
        raise ValueError("tokens/region_xy/prompt_xy must be [B,N,D]/[B,N,2]/[B,P,2]")
    if tokens.shape[:2] != region_xy.shape[:2] or prompt_xy.shape[:2] != valid.shape:
        raise ValueError("online prompt tensor shapes are incompatible")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    distance = torch.cdist(prompt_xy.float(), region_xy.float()).to(tokens.dtype)
    hard_indices = distance.argmin(-1)
    hard = torch.nn.functional.one_hot(hard_indices, tokens.shape[1]).to(tokens.dtype)
    soft = torch.softmax(-distance / float(temperature), dim=-1)
    weights = hard + soft - soft.detach()
    weights = weights * valid.unsqueeze(-1).to(weights.dtype)
    prompt_tokens = torch.einsum("bpk,bkd->bpd", weights, tokens)
    hard_indices = hard_indices.masked_fill(~valid, -1)
    safe = hard_indices.clamp_min(0)
    membership_count = torch.zeros(
        tokens.shape[0], tokens.shape[1], dtype=torch.long, device=tokens.device
    )
    membership_count.scatter_add_(1, safe, valid.long())
    membership = membership_count > 0
    return prompt_tokens, hard_indices, membership, weights, soft * valid.unsqueeze(-1).to(soft.dtype)


def prompt_region_membership(
    region_xy: torch.Tensor, prompt_xy: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map prompt coordinates to nearest region slots without a token path."""
    if region_xy.ndim != 3 or prompt_xy.ndim != 3 or prompt_xy.shape[:2] != valid.shape:
        raise ValueError("region/prompt geometry shapes are incompatible")
    indices = torch.cdist(prompt_xy.float(), region_xy.float()).argmin(-1)
    indices = indices.masked_fill(~valid, -1)
    membership_count = torch.zeros(
        region_xy.shape[:2], dtype=torch.long, device=region_xy.device
    )
    membership_count.scatter_add_(1, indices.clamp_min(0), valid.long())
    return indices, membership_count > 0


class FrozenRegionGeometryTeacher(nn.Module):
    """Frozen old-joint Phase2 heads evaluated on shared backbone features."""

    def __init__(self, phase2: DeepRegionEncoder, checkpoint: str | Path):
        super().__init__()
        self.embedding = copy.deepcopy(phase2.embedding)
        self.assignment = copy.deepcopy(phase2.assignment)
        self.register_buffer("spatial_anchors", phase2.spatial_anchors.detach().clone(), persistent=True)
        self.spatial_temperature = float(phase2.spatial_temperature)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"]
        embedding_state = {
            key.removeprefix("phase2.embedding."): value
            for key, value in state.items() if key.startswith("phase2.embedding.")
        }
        assignment_state = {
            key.removeprefix("phase2.assignment."): value
            for key, value in state.items() if key.startswith("phase2.assignment.")
        }
        if not embedding_state or not assignment_state:
            raise ValueError(f"{checkpoint} has no Phase2 embedding/assignment heads")
        self.embedding.load_state_dict(embedding_state, strict=True)
        self.assignment.load_state_dict(assignment_state, strict=True)
        self.requires_grad_(False)
        self.eval()
        self.transfer = {
            "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
            "embedding_tensors": len(embedding_state), "assignment_tensors": len(assignment_state),
        }

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, features: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.embedding(features.detach())
        assignment = spatial_region_assignment(
            embedding, self.assignment, self.spatial_anchors, self.spatial_temperature
        )
        xy, area = assignment_geometry(assignment)
        positive_slots, positive_regions = prompt_region_membership(
            xy, batch["positive_xy"], batch["positive_mask"]
        )
        negative_slots, negative_regions = prompt_region_membership(
            xy, batch["negative_xy"], batch["negative_mask"]
        )
        return {
            "geometry_teacher_region_xy": xy,
            "geometry_teacher_region_area": area,
            "geometry_teacher_positive_slot_indices": positive_slots,
            "geometry_teacher_negative_slot_indices": negative_slots,
            "geometry_teacher_prompt_conflicts": positive_regions & negative_regions,
        }


class FrozenPromptTeacher(nn.Module):
    """Unregistered Phase5 snapshot used while selected prompt layers train."""

    def __init__(self, prompt: PromptRegionModel):
        super().__init__()
        self.model = copy.deepcopy(prompt)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch)


class ParentContextAdapter(nn.Module):
    """Zero-gated middle/coarse parent context for online fine tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, fine: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        required = (
            "middle_tokens", "coarse_tokens",
            "fine_middle_edge_index", "fine_middle_edge_weight",
            "middle_coarse_edge_index", "middle_coarse_edge_weight",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"parent context batch misses {missing}")
        middle = batch["middle_tokens"].to(dtype=fine.dtype)
        coarse = batch["coarse_tokens"].to(dtype=fine.dtype)
        middle_for_fine = gather_parents(
            middle, batch["fine_middle_edge_index"], batch["fine_middle_edge_weight"]
        )
        coarse_for_middle = gather_parents(
            coarse, batch["middle_coarse_edge_index"], batch["middle_coarse_edge_weight"]
        )
        coarse_for_fine = gather_parents(
            coarse_for_middle, batch["fine_middle_edge_index"], batch["fine_middle_edge_weight"]
        )
        residual = self.fuse(torch.cat([middle_for_fine, coarse_for_fine], dim=-1))
        return fine + torch.tanh(self.gate).to(fine.dtype) * residual


@torch.no_grad()
def online_region_binary_target(
    assignment: torch.Tensor,
    pixel_gt: torch.Tensor,
    target_class: torch.Tensor,
    active: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached majority-class region targets from the current assignment."""
    if assignment.ndim != 4 or pixel_gt.ndim != 3:
        raise ValueError("assignment and pixel_gt must be [B,K,H,W] and [B,H,W]")
    if assignment.shape[0] != pixel_gt.shape[0] or assignment.shape[-2:] != pixel_gt.shape[-2:]:
        raise ValueError("assignment and pixel_gt spatial shapes do not match")
    if active.shape != assignment.shape[:2] or target_class.shape != (assignment.shape[0],):
        raise ValueError("active/target_class shapes do not match assignment")
    classes = int(num_classes); ignore = int(ignore_index)
    hard = assignment.detach().argmax(1)
    target = torch.full(active.shape, ignore, dtype=torch.long, device=assignment.device)
    purity = torch.zeros(active.shape, dtype=torch.float32, device=assignment.device)
    for batch_index in range(assignment.shape[0]):
        valid_pixel = pixel_gt[batch_index] != ignore
        if not valid_pixel.any():
            continue
        labels = pixel_gt[batch_index][valid_pixel]
        if int(labels.min()) < 0 or int(labels.max()) >= classes:
            raise ValueError("pixel_gt contains a non-ignore class outside num_classes")
        votes = torch.bincount(
            hard[batch_index][valid_pixel] * classes + labels,
            minlength=assignment.shape[1] * classes,
        ).reshape(assignment.shape[1], classes)
        mass = votes.sum(1); present = active[batch_index] & (mass > 0)
        majority_votes, majority_class = votes.max(1)
        target[batch_index, present] = (
            majority_class[present] == target_class[batch_index]
        ).long()
        purity[batch_index, present] = majority_votes[present].float() / mass[present].float()
    return target, purity


class JointPromptMaskModel(nn.Module):
    def __init__(
        self,
        phase2: DeepRegionEncoder,
        cell: CellToRegionAttention,
        prompt: PromptRegionModel,
        region_dim: int = 256,
        graph_heads: int = 4,
        graph_layers: int = 2,
        graph_neighbours: int = 6,
        graph_dropout: float = 0.1,
        residual_limit: float = 1.0,
        train_phase2_embedding: bool = True,
        train_phase2_assignment: bool = True,
        train_cell: bool = False,
        train_prompt: bool = False,
        train_decoder: bool = True,
        train_parent_context: bool = False,
        train_backbone_layer4: bool = False,
        num_classes: int = 12,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.phase2 = phase2; self.cell = cell
        self.train_phase2_embedding = bool(train_phase2_embedding)
        self.train_phase2_assignment = bool(train_phase2_assignment)
        self.train_cell = bool(train_cell); self.train_prompt = bool(train_prompt)
        self.train_decoder = bool(train_decoder); self.train_parent_context = bool(train_parent_context)
        self.train_backbone_layer4 = bool(train_backbone_layer4)
        self.num_classes = int(num_classes); self.ignore_index = int(ignore_index)

        self.phase2.requires_grad_(False)
        if self.train_backbone_layer4:
            if not hasattr(self.phase2.backbone, "layer4"):
                raise ValueError("train_backbone_layer4 requires phase2.backbone.layer4")
            self.phase2.backbone.layer4.requires_grad_(True)
        self.phase2.embedding.requires_grad_(self.train_phase2_embedding)
        self.phase2.assignment.requires_grad_(self.train_phase2_assignment)
        self.cell.requires_grad_(self.train_cell)
        self.decoder = ContextAwareMaskDecoder(
            prompt, region_dim, graph_heads, graph_layers, graph_neighbours, graph_dropout,
            freeze_prompt_encoder=not self.train_prompt, residual_limit=residual_limit,
        )
        self.decoder.requires_grad_(self.train_decoder)
        self.decoder.prompt_model.requires_grad_(False)
        if self.train_prompt:
            self.decoder.prompt_model.matcher.requires_grad_(True)
            self.decoder.prompt_model.task_projection.requires_grad_(True)
            self.decoder.prompt_model.set_pool.encoder.layers[-1].requires_grad_(True)
        self.parent_context = ParentContextAdapter(region_dim) if self.train_parent_context else None

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen pretrained modules stay deterministic; gradients can still
        # flow through them to trainable upstream inputs.
        self.phase2.backbone.eval(); self.phase2.embedding.eval(); self.phase2.semantic.eval()
        if not self.train_cell:
            self.cell.eval()
        # Keep the pretrained prompt path deterministic even when selected
        # layers receive gradients; eval mode does not disable autograd.
        self.decoder.prompt_model.eval()
        return self

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        geometry_teacher: FrozenRegionGeometryTeacher | None = None,
        prompt_teacher: FrozenPromptTeacher | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.phase2.extract_backbone_features(batch["image"])
        phase2 = self.phase2.forward_from_features(
            features, batch["image"].shape[-2:], return_full_assignment=True, return_tokens=True
        )
        assignment_low = phase2["assignment_low"]
        cell_mass = sample_assignment_at_cells(assignment_low, batch["cells"])
        cell = self.cell(
            phase2["region_tokens"], cell_mass, batch["cells"], batch["cell_valid"], batch["total_cell_count"].float()
        )
        xy, area = assignment_geometry(assignment_low)
        contextual_tokens = (
            self.parent_context(cell["fused_tokens"], batch)
            if self.parent_context is not None else cell["fused_tokens"]
        )
        positive_tokens, positive_slots, positive_regions, positive_weights, positive_soft = remap_prompt_tokens(
            contextual_tokens, xy, batch["positive_xy"], batch["positive_mask"]
        )
        negative_tokens, negative_slots, negative_regions, negative_weights, negative_soft = remap_prompt_tokens(
            contextual_tokens, xy, batch["negative_xy"], batch["negative_mask"]
        )
        online_target, online_purity = online_region_binary_target(
            phase2["assignment"], batch["pixel_gt"], batch["target_class"], batch["fine_active"],
            self.num_classes, self.ignore_index,
        )
        prompt_batch = dict(batch)
        prompt_batch.update({
            "fine_tokens": contextual_tokens,
            "region_xy": xy,
            "region_area": area,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
        })
        decoded = self.decoder(prompt_batch)
        if self.train_prompt and prompt_teacher is None:
            raise ValueError("train_prompt=true requires an explicit frozen prompt teacher")
        if not self.train_prompt and prompt_teacher is not None:
            raise ValueError("frozen prompt teacher is only valid when train_prompt=true")
        with torch.no_grad():
            cached_teacher = (
                prompt_teacher(batch) if prompt_teacher is not None
                else self.decoder.prompt_model(batch)
            )
        pixel = project_region_probabilities(phase2["assignment"], decoded["logits"])
        output = {
            **decoded, **pixel,
            "assignment": phase2["assignment"], "assignment_low": assignment_low,
            "online_region_tokens": phase2["region_tokens"], "online_fused_tokens": cell["fused_tokens"],
            "online_contextual_tokens": contextual_tokens,
            "online_region_xy": xy, "online_region_area": area,
            "online_binary_target": online_target, "online_region_purity": online_purity,
            "online_positive_slot_indices": positive_slots,
            "online_negative_slot_indices": negative_slots,
            "online_prompted_regions": positive_regions,
            "online_negative_prompted_regions": negative_regions,
            "online_all_prompted_regions": positive_regions | negative_regions,
            "online_prompt_conflicts": positive_regions & negative_regions,
            "online_positive_prompt_weights": positive_weights,
            "online_negative_prompt_weights": negative_weights,
            "online_positive_prompt_soft_weights": positive_soft,
            "online_negative_prompt_soft_weights": negative_soft,
            "teacher_logits": cached_teacher["logits"],
            "teacher_task_token": cached_teacher["task_token"],
            "semantic_logits": phase2["semantic_logits"], "region_cell_count": cell["region_cell_count"],
        }
        if self.parent_context is not None:
            output["parent_context_gate"] = torch.tanh(self.parent_context.gate)
        if geometry_teacher is not None:
            output.update(geometry_teacher(features, batch))
        return output


def load_joint_components(
    phase2_config: dict,
    phase2_checkpoint: str | Path,
    cell_checkpoint: str | Path,
    phase5_checkpoint: str | Path,
    phase5_config: dict,
    cell_dim: int,
) -> tuple[DeepRegionEncoder, CellToRegionAttention, PromptRegionModel, dict]:
    region_dim = int(phase2_config["model"]["embedding_dim"])
    phase2 = DeepRegionEncoder(
        phase2_config, len(phase2_config["data"]["class_map"]),
        int(phase2_config["model"]["num_regions"]), region_dim,
    )
    p2_payload = torch.load(phase2_checkpoint, map_location="cpu", weights_only=False)
    p2_missing, p2_unexpected = phase2.load_state_dict(p2_payload["model"], strict=True)
    cell = CellToRegionAttention(region_dim, int(cell_dim))
    cell_payload = torch.load(cell_checkpoint, map_location="cpu", weights_only=False)
    if cell_payload.get("cell") is None:
        raise ValueError(f"{cell_checkpoint} has no cell module")
    cell_missing, cell_unexpected = cell.load_state_dict(cell_payload["cell"], strict=True)
    prompt_cfg = phase5_config["model"]
    prompt = PromptRegionModel(
        int(prompt_cfg["region_dim"]), int(prompt_cfg["heads"]),
        int(prompt_cfg["set_layers"]), float(prompt_cfg["dropout"]),
    )
    prompt_payload = torch.load(phase5_checkpoint, map_location="cpu", weights_only=False)
    prompt_missing, prompt_unexpected = prompt.load_state_dict(prompt_payload["model"], strict=True)
    transfer = {
        "phase2": {"checkpoint": str(phase2_checkpoint), "epoch": int(p2_payload["epoch"]), "tensors": len(p2_payload["model"]), "missing": list(p2_missing), "unexpected": list(p2_unexpected)},
        "cell": {"checkpoint": str(cell_checkpoint), "epoch": int(cell_payload["epoch"]), "tensors": len(cell_payload["cell"]), "missing": list(cell_missing), "unexpected": list(cell_unexpected)},
        "prompt": {"checkpoint": str(phase5_checkpoint), "epoch": int(prompt_payload["epoch"]), "tensors": len(prompt_payload["model"]), "missing": list(prompt_missing), "unexpected": list(prompt_unexpected)},
    }
    return phase2, cell, prompt, transfer
