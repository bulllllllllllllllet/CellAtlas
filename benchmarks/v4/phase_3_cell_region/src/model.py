from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_assignment_at_cells(assignment_low: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B,K,H,W] soft assignments at normalized cell x/y coordinates."""
    if assignment_low.ndim != 4 or cells.ndim != 3 or cells.shape[0] != assignment_low.shape[0] or cells.shape[-1] < 2:
        raise ValueError("expected assignment [B,K,H,W] and cells [B,N,>=2]")
    if cells.shape[1] == 0: return assignment_low.new_zeros((assignment_low.shape[0],assignment_low.shape[1],0))
    grid=(cells[...,:2]*2-1).unsqueeze(1)
    return F.grid_sample(assignment_low,grid,mode="bilinear",padding_mode="border",align_corners=False).squeeze(2)


class CellToRegionAttention(torch.nn.Module):
    """Prompt-independent cell pooling, constrained by each cell's soft region mass."""
    def __init__(self, region_dim: int = 256, cell_dim: int = 68, hidden_dim: int = 128):
        super().__init__()
        self.cell = torch.nn.Sequential(torch.nn.LayerNorm(cell_dim),torch.nn.Linear(cell_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, region_dim))
        self.query = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.key = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.value = torch.nn.Linear(region_dim, region_dim, bias=False)
        self.fuse = torch.nn.Sequential(torch.nn.Linear(region_dim * 2 + 1, region_dim), torch.nn.ReLU(), torch.nn.LayerNorm(region_dim))

    def forward(self, region_tokens: torch.Tensor, assignment_at_cells: torch.Tensor, cells: torch.Tensor, cell_valid: torch.Tensor, total_cell_count: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # region_tokens [B,K,D], assignment_at_cells [B,K,N], cells [B,N,D_cell].
        if assignment_at_cells.shape[:2] != region_tokens.shape[:2] or cells.shape[:2] != cell_valid.shape:
            raise ValueError("incompatible region/cell shapes")
        encoded=self.cell(cells); scale=region_tokens.shape[-1] ** -0.5
        logits=torch.einsum("bkd,bnd->bkn",self.query(region_tokens),self.key(encoded))*scale
        logits=logits + assignment_at_cells.clamp_min(1e-8).log()
        logits=logits.masked_fill(~cell_valid[:,None,:],float("-inf"))
        has_cells=cell_valid.any(1); safe_logits=torch.where(has_cells[:,None,None],logits,torch.zeros_like(logits))
        weights=safe_logits.softmax(-1)*cell_valid[:,None,:]
        context=torch.einsum("bkn,bnd->bkd",weights,self.value(encoded))
        # Hybrid caches retain at most max_cells embeddings, but also store the
        # true count.  Distribute that total over regions using sampled soft
        # assignment mass, rather than copying the selected count to every region.
        selected=cell_valid.float().sum(1)
        if total_cell_count is None: total_cell_count=selected
        if total_cell_count.shape!=(cells.shape[0],): raise ValueError("total_cell_count must have shape [B]")
        scale_factor=torch.where(selected>0,total_cell_count.to(selected).clamp_min(0)/selected.clamp_min(1),torch.zeros_like(selected))
        normalized_mass=assignment_at_cells/assignment_at_cells.sum(1,keepdim=True).clamp_min(1e-8)
        region_cell_count=(normalized_mass*cell_valid[:,None,:]).sum(-1)*scale_factor[:,None]
        density=region_cell_count.log1p().unsqueeze(-1)
        fused=self.fuse(torch.cat((region_tokens,context,density),-1))
        return {"fused_tokens":fused,"cell_context":context,"attention":weights,"has_cells":has_cells,"region_cell_count":region_cell_count}
