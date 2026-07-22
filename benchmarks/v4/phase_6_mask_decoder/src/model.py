"""Sparse local graph refinement and soft region-to-pixel projection."""
from __future__ import annotations

import torch
import torch.nn as nn

from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel


def knn_adjacency(xy: torch.Tensor, active: torch.Tensor, k: int) -> torch.Tensor:
    """Return directed KNN plus self edges as ``[B,N,N]`` boolean adjacency."""
    if xy.ndim != 3 or xy.shape[-1] != 2 or active.shape != xy.shape[:2]:
        raise ValueError("xy must be [B,N,2] and active must be [B,N]")
    batch, regions, _ = xy.shape
    if regions < 1:
        raise ValueError("at least one region is required")
    neighbours = min(max(int(k), 0), max(regions - 1, 0))
    adjacency = torch.zeros(batch, regions, regions, dtype=torch.bool, device=xy.device)
    eye = torch.eye(regions, dtype=torch.bool, device=xy.device)[None]
    adjacency |= eye
    if neighbours:
        distance = torch.cdist(xy.float(), xy.float())
        candidates = active[:, None, :].expand(-1, regions, -1) & ~eye
        distance = distance.masked_fill(~candidates, torch.inf)
        values, indices = distance.topk(neighbours, dim=-1, largest=False)
        adjacency.scatter_(2, indices, torch.isfinite(values))
    adjacency &= active[:, :, None] & active[:, None, :]
    # Inactive query rows retain a self edge so attention never receives an all-masked row.
    adjacency |= eye & ~active[:, :, None]
    return adjacency


class LocalGraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = int(heads)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, regions, _ = tokens.shape
        blocked = ~adjacency[:, None].expand(-1, self.heads, -1, -1)
        blocked = blocked.reshape(batch * self.heads, regions, regions)
        normalized = self.norm1(tokens)
        context, _ = self.attention(
            normalized, normalized, normalized, attn_mask=blocked, need_weights=False
        )
        tokens = tokens + context
        return tokens + self.ffn(self.norm2(tokens))


class ContextAwareMaskDecoder(nn.Module):
    """Refine frozen Phase-5 logits using a local region graph.

    The delta head is zero-initialized, making the initial decoder exactly
    equivalent to the supplied Phase-5 model at the region-logit level.
    """

    def __init__(
        self,
        prompt_model: PromptRegionModel,
        dim: int = 256,
        heads: int = 4,
        graph_layers: int = 2,
        neighbours: int = 6,
        dropout: float = 0.1,
        freeze_prompt_encoder: bool = True,
        residual_limit: float = 1.0,
    ):
        super().__init__()
        if int(prompt_model.dim) != int(dim):
            raise ValueError(f"prompt dim {prompt_model.dim} != decoder dim {dim}")
        self.prompt_model = prompt_model
        self.neighbours = int(neighbours)
        self.freeze_prompt_encoder = bool(freeze_prompt_encoder)
        self.residual_limit = float(residual_limit)
        if self.residual_limit <= 0:
            raise ValueError("residual_limit must be positive")
        if self.freeze_prompt_encoder:
            self.prompt_model.requires_grad_(False)
        self.logit_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.geometry_embedding = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_projection = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            LocalGraphBlock(dim, heads, dropout) for _ in range(int(graph_layers))
        )
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_prompt_encoder:
            self.prompt_model.eval()
        return self

    def refine(
        self,
        batch: dict[str, torch.Tensor],
        prompt: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Refine region logits produced locally or from a reusable WSI prompt task."""
        active = batch["fine_active"].bool()
        adjacency = knn_adjacency(batch["region_xy"], active, self.neighbours)
        geometry = torch.cat([batch["region_xy"], batch["region_area"].unsqueeze(-1)], dim=-1)
        hidden = (
            prompt["task_aware_tokens"]
            + self.logit_embedding(prompt["logits"].unsqueeze(-1))
            + self.geometry_embedding(geometry)
            + self.task_projection(prompt["task_token"])[:, None]
        )
        hidden = self.input_norm(hidden)
        hidden = hidden * active.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, adjacency)
            hidden = hidden * active.unsqueeze(-1).to(hidden.dtype)
        raw_delta = self.residual_limit * torch.tanh(
            self.delta_head(hidden).squeeze(-1) / self.residual_limit
        )
        active_float = active.to(hidden.dtype)
        mean_delta = (raw_delta * active_float).sum(1, keepdim=True) / active_float.sum(1, keepdim=True).clamp_min(1)
        delta = (raw_delta - mean_delta) * active_float
        logits = prompt["logits"] + delta
        return {
            **prompt,
            "initial_logits": prompt["logits"],
            "logits": logits,
            "logit_delta": delta,
            "raw_logit_delta": raw_delta * active_float,
            "decoder_tokens": hidden,
            "adjacency": adjacency,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Frozen prompt parameters must still permit gradients with respect to
        # online Phase-2/3 input tokens during end-to-end pixel training.
        return self.refine(batch, self.prompt_model(batch))


def project_region_probabilities(
    assignment: torch.Tensor, region_logits: torch.Tensor, eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    """Project region probabilities through soft assignments to pixel space."""
    if assignment.ndim != 4 or region_logits.ndim != 2:
        raise ValueError("assignment must be [B,K,H,W], logits must be [B,K]")
    if assignment.shape[:2] != region_logits.shape:
        raise ValueError("assignment slots and region logits do not match")
    normalized = assignment / assignment.sum(1, keepdim=True).clamp_min(eps)
    probability = torch.einsum("bkhw,bk->bhw", normalized, region_logits.sigmoid())
    # BF16 cannot represent 1-eps for eps=1e-6, so clamping before promotion
    # can leave exact ones and make torch.logit return +inf.
    probability = probability.float().clamp(eps, 1.0 - eps)
    return {"pixel_probability": probability, "pixel_logits": torch.logit(probability)}


def load_phase5_prompt_model(
    checkpoint: str, dim: int, heads: int, set_layers: int, dropout: float
) -> tuple[PromptRegionModel, dict]:
    model = PromptRegionModel(dim, heads, set_layers, dropout)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Phase5 load mismatch missing={missing} unexpected={unexpected}")
    return model, {
        "checkpoint": str(checkpoint),
        "source_epoch": int(payload["epoch"]),
        "loaded_tensor_count": len(payload["model"]),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
