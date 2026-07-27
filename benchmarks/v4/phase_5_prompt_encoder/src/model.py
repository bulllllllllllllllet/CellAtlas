"""Prompt set encoder and prompt-conditioned fine-region matcher."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptSetPool(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(dim))
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not valid.any(1).all():
            raise ValueError("each prompt set must contain at least one valid token")
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=~valid, need_weights=False)
        return pooled[:, 0]


class PromptRegionModel(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, set_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.prompt_pool_variant = "set_encoder"
        self.region_norm = nn.LayerNorm(dim)
        self.prompt_token_norm = nn.LayerNorm(dim)
        self.prompt_geometry = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.region_geometry = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.sign_embedding = nn.Embedding(2, dim)
        self.size_embedding = nn.Embedding(3, dim)
        self.set_pool = PromptSetPool(dim, heads, set_layers, dropout)
        self.task_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )
        self.region_prompt_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.task_projection = nn.Linear(dim, dim)
        self.task_aware_norm = nn.LayerNorm(dim)
        self.matcher = nn.Sequential(
            nn.Linear(dim * 4 + 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _encode_prompt_set(
        self,
        tokens: torch.Tensor,
        xy: torch.Tensor,
        valid: torch.Tensor,
        sign_id: int,
        size_id: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.prompt_token_norm(tokens) + self.prompt_geometry(xy)
        embedded = embedded + self.sign_embedding.weight[sign_id] + self.size_embedding(size_id)[:, None]
        embedded = embedded * valid.unsqueeze(-1).to(embedded.dtype)
        if self.prompt_pool_variant == "set_encoder":
            return self.set_pool(embedded, valid)
        if self.prompt_pool_variant == "mean_prototype":
            if not valid.any(1).all():
                raise ValueError("each prompt set must contain at least one valid token")
            count = valid.sum(1, keepdim=True).clamp_min(1).to(embedded.dtype)
            return embedded.sum(1) / count
        raise ValueError(f"unsupported prompt_pool_variant={self.prompt_pool_variant}")

    def encode_prompt_task(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode positive/negative prompt sets independently of candidate regions."""
        q_pos = self._encode_prompt_set(
            batch["positive_tokens"], batch["positive_xy"], batch["positive_mask"], 0, batch["prompt_size_id"]
        )
        q_neg = self._encode_prompt_set(
            batch["negative_tokens"], batch["negative_xy"], batch["negative_mask"], 1, batch["prompt_size_id"]
        )
        task = self.task_mlp(torch.cat([q_pos, q_neg, q_pos - q_neg], dim=-1))
        return {"task_token": task, "q_positive": q_pos, "q_negative": q_neg}

    def match_regions(
        self,
        batch: dict[str, torch.Tensor],
        prompt_task: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Match candidate regions against a reusable encoded prompt task."""
        region = self.region_norm(batch["fine_tokens"])
        q_pos = prompt_task["q_positive"]
        q_neg = prompt_task["q_negative"]
        task = prompt_task["task_token"]
        if q_pos.shape != q_neg.shape or q_pos.shape != task.shape:
            raise ValueError("prompt task tensors must share shape [B,D]")
        if q_pos.ndim != 2 or q_pos.shape[0] != region.shape[0] or q_pos.shape[1] != self.dim:
            raise ValueError("prompt task batch/dimension does not match candidate regions")
        sign_tokens = torch.stack([q_pos, q_neg], dim=1)
        context, _ = self.region_prompt_attention(region, sign_tokens, sign_tokens, need_weights=False)
        sim_pos = F.cosine_similarity(region, q_pos[:, None], dim=-1)
        sim_neg = F.cosine_similarity(region, q_neg[:, None], dim=-1)
        geometry = self.region_geometry(
            torch.cat([batch["region_xy"], batch["region_area"].unsqueeze(-1)], dim=-1)
        )
        task_expanded = task[:, None].expand(-1, region.shape[1], -1)
        task_aware = self.task_aware_norm(region + context + self.task_projection(task)[:, None])
        features = torch.cat(
            [task_aware, context, geometry, task_expanded, sim_pos.unsqueeze(-1), sim_neg.unsqueeze(-1)], dim=-1
        )
        logits = self.matcher(features).squeeze(-1)
        return {
            "logits": logits,
            "task_token": task,
            "task_aware_tokens": task_aware,
            "q_positive": q_pos,
            "q_negative": q_neg,
            "similarity_difference": sim_pos - sim_neg,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.match_regions(batch, self.encode_prompt_task(batch))


def load_fine_norm_from_phase4(model: PromptRegionModel, checkpoint: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    expected = {"fine_norm.weight", "fine_norm.bias"}
    missing = sorted(expected - set(state))
    if missing:
        raise RuntimeError(f"Phase4 checkpoint missing fine norm tensors: {missing}")
    with torch.no_grad():
        model.region_norm.weight.copy_(state["fine_norm.weight"])
        model.region_norm.bias.copy_(state["fine_norm.bias"])
    return {
        "checkpoint": str(checkpoint),
        "loaded": ["fine_norm.weight->region_norm.weight", "fine_norm.bias->region_norm.bias"],
        "intentionally_not_loaded": sorted(set(state) - expected),
    }
