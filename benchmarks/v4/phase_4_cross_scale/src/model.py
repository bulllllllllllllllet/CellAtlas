"""Phase-4 cross-scale models over frozen multi-scale region tokens."""
from __future__ import annotations

import torch
import torch.nn as nn


def gather_parents(parent: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Weighted sum of parent tokens for each child.

    parent: [B, P, D]
    edge_index: [B, C, K] with -1 padding
    edge_weight: [B, C, K]
    returns: [B, C, D]
    """
    batch, n_parent, dim = parent.shape
    _, n_child, top_k = edge_index.shape
    safe_index = edge_index.clamp_min(0)
    batch_ids = torch.arange(batch, device=parent.device)[:, None, None].expand(-1, n_child, top_k)
    gathered = parent[batch_ids, safe_index]
    # Match AMP dtype: edge_weight is float32 on load, parent may be bf16/fp16.
    weight = edge_weight.to(dtype=parent.dtype).unsqueeze(-1) * (edge_index >= 0).unsqueeze(-1).to(parent.dtype)
    return (gathered * weight).sum(2)


def gather_children(child: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, n_parent: int) -> torch.Tensor:
    """Aggregate child tokens onto parents using child→parent edges."""
    batch, n_child, dim = child.shape
    top_k = edge_index.shape[-1]
    out = child.new_zeros(batch, n_parent, dim)
    mass = child.new_zeros(batch, n_parent, 1)
    safe_index = edge_index.clamp_min(0)
    valid = edge_index >= 0
    edge_weight = edge_weight.to(dtype=child.dtype)
    for k in range(top_k):
        idx = safe_index[:, :, k]
        w = edge_weight[:, :, k] * valid[:, :, k].to(child.dtype)
        src = child * w.unsqueeze(-1)
        out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, dim), src)
        mass.scatter_add_(1, idx.unsqueeze(-1), w.unsqueeze(-1))
    return out / mass.clamp_min(1e-6)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMessage(nn.Module):
    """Residual message with LN + FFN."""

    def __init__(self, dim: int):
        super().__init__()
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward_from_parents(
        self,
        child: torch.Tensor,
        parent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        context = gather_parents(self.v(parent), edge_index, edge_weight)
        msg = self.proj(context)
        x = self.norm1(child + msg)
        return self.norm2(x + self.ff(x))

    def forward_from_children(
        self,
        parent: torch.Tensor,
        child: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        context = gather_children(self.v(child), edge_index, edge_weight, n_parent=parent.shape[1])
        msg = self.proj(context)
        x = self.norm1(parent + msg)
        return self.norm2(x + self.ff(x))


class HierarchicalBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up_fm = SparseMessage(dim)
        self.up_mc = SparseMessage(dim)
        self.down_cm = SparseMessage(dim)
        self.down_mf = SparseMessage(dim)

    def forward(self, fine, middle, coarse, fm_index, fm_weight, mc_index, mc_weight):
        middle = self.up_fm.forward_from_children(middle, fine, fm_index, fm_weight)
        coarse = self.up_mc.forward_from_children(coarse, middle, mc_index, mc_weight)
        middle = self.down_cm.forward_from_parents(middle, coarse, mc_index, mc_weight)
        fine = self.down_mf.forward_from_parents(fine, middle, fm_index, fm_weight)
        return fine, middle, coarse


class CrossScaleModel(nn.Module):
    """Unified Phase-4 variants over cached tokens.

    Only constructs parameters required by the selected variant so DDP does not
    see unused parameters.
    """

    VARIANTS = (
        "fine_only",
        "middle_only",
        "coarse_only",
        "naive_concat",
        "hierarchical_up",
        "hierarchical_bidir",
    )

    def __init__(self, variant: str, dim: int = 256, num_classes: int = 12, num_blocks: int = 1):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.dim = dim
        # always keep a fine-scale bias so fine_only is not a pure linear probe on frozen tokens alone
        self.fine_norm = nn.LayerNorm(dim)
        self.scale_embed = None
        self.blocks = None
        self.fuse = None
        if variant != "fine_only":
            self.scale_embed = nn.Embedding(3, dim)
        if variant in ("middle_only", "coarse_only"):
            self.fuse = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
            )
        elif variant == "naive_concat":
            self.fuse = nn.Sequential(
                nn.Linear(dim * 3, dim * 2),
                nn.GELU(),
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
            )
        elif variant in ("hierarchical_up", "hierarchical_bidir"):
            self.blocks = nn.ModuleList([HierarchicalBlock(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, num_classes)

    def _add_scale(self, fine, middle, coarse):
        if self.scale_embed is None:
            return fine, middle, coarse
        fine = fine + self.scale_embed.weight[0]
        middle = middle + self.scale_embed.weight[1]
        coarse = coarse + self.scale_embed.weight[2]
        return fine, middle, coarse

    def _compose_coarse_for_fine(self, middle, coarse, fm_index, fm_weight, mc_index, mc_weight):
        coarse_for_middle = gather_parents(coarse, mc_index, mc_weight)
        return gather_parents(coarse_for_middle, fm_index, fm_weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fine = batch["fine_tokens"]
        middle = batch["middle_tokens"]
        coarse = batch["coarse_tokens"]
        fm_index = batch["fine_middle_edge_index"]
        fm_weight = batch["fine_middle_edge_weight"]
        mc_index = batch["middle_coarse_edge_index"]
        mc_weight = batch["middle_coarse_edge_weight"]
        fine, middle, coarse = self._add_scale(fine, middle, coarse)

        if self.variant == "fine_only":
            tokens = self.fine_norm(fine)
        elif self.variant == "middle_only":
            mid_ctx = gather_parents(middle, fm_index, fm_weight)
            tokens = self.fuse(torch.cat([self.fine_norm(fine), mid_ctx], dim=-1))
        elif self.variant == "coarse_only":
            coarse_ctx = self._compose_coarse_for_fine(middle, coarse, fm_index, fm_weight, mc_index, mc_weight)
            tokens = self.fuse(torch.cat([self.fine_norm(fine), coarse_ctx], dim=-1))
        elif self.variant == "naive_concat":
            mid_ctx = gather_parents(middle, fm_index, fm_weight)
            coarse_ctx = self._compose_coarse_for_fine(middle, coarse, fm_index, fm_weight, mc_index, mc_weight)
            tokens = self.fuse(torch.cat([self.fine_norm(fine), mid_ctx, coarse_ctx], dim=-1))
        elif self.variant == "hierarchical_up":
            for block in self.blocks:
                middle = block.up_fm.forward_from_children(middle, fine, fm_index, fm_weight)
                coarse = block.up_mc.forward_from_children(coarse, middle, mc_index, mc_weight)
                middle = block.down_cm.forward_from_parents(middle, coarse, mc_index, mc_weight)
                fine = block.down_mf.forward_from_parents(fine, middle, fm_index, fm_weight)
            tokens = self.fine_norm(fine)
        else:
            for block in self.blocks:
                fine, middle, coarse = block(fine, middle, coarse, fm_index, fm_weight, mc_index, mc_weight)
            tokens = self.fine_norm(fine)
        return self.head(tokens)
