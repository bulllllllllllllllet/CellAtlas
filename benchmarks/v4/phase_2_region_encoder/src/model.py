from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmarks.v4.phase_1_multiscale.src.model import build_model as build_deeplab


def spatial_region_assignment(
    embedding: torch.Tensor,
    assignment_head: torch.nn.Module,
    spatial_anchors: torch.Tensor,
    spatial_temperature: float,
) -> torch.Tensor:
    """Apply the learned assignment head plus its fixed spatial prior."""
    height, width = embedding.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=embedding.device, dtype=embedding.dtype),
        torch.linspace(-1, 1, width, device=embedding.device, dtype=embedding.dtype),
        indexing="ij",
    )
    coords = torch.stack((yy, xx))
    anchors = spatial_anchors.to(device=embedding.device, dtype=embedding.dtype)[:, :, None, None]
    spatial_logits = -((coords[None] - anchors) ** 2).sum(1) / float(spatial_temperature)
    return (assignment_head(embedding) + spatial_logits).softmax(1)


class DeepRegionEncoder(torch.nn.Module):
    """Pixel features, soft region assignments, and pooled region tokens.

    Assignment is deliberately soft during training.  ``region_tokens`` is the
    area-normalised weighted average required by later cell and graph modules.
    """
    def __init__(self, config: dict, num_classes: int, num_regions: int, embedding_dim: int = 256, input_channels: int = 3):
        super().__init__()
        base = build_deeplab(config, num_classes, input_channels=input_channels)
        self.backbone = base.backbone
        self.embedding = torch.nn.Sequential(
            torch.nn.Conv2d(2048, embedding_dim, 1, bias=False),
            torch.nn.BatchNorm2d(embedding_dim),
            torch.nn.ReLU(inplace=True),
        )
        grid_size=int(round(num_regions**0.5))
        if grid_size*grid_size != num_regions: raise ValueError(f"num_regions must be a square number, got {num_regions}")
        self.assignment = torch.nn.Conv2d(embedding_dim, num_regions, 1)
        torch.nn.init.zeros_(self.assignment.weight); torch.nn.init.zeros_(self.assignment.bias)
        anchors=torch.tensor([(-1+(row+.5)*2/grid_size,-1+(col+.5)*2/grid_size) for row in range(grid_size) for col in range(grid_size)],dtype=torch.float32)
        self.register_buffer("spatial_anchors",anchors,persistent=True)
        self.spatial_temperature=float(config["model"]["spatial_temperature"])
        # Retain the selected Phase 1 ASPP/classifier instead of discarding its
        # learned tissue segmentation weights.
        self.semantic = base.classifier
        self.num_regions = num_regions

    def forward_from_features(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int],
        return_full_assignment: bool = True,
        return_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run region heads on shared backbone features."""
        embedding_low = self.embedding(features)
        assignment_low = spatial_region_assignment(
            embedding_low, self.assignment, self.spatial_anchors, self.spatial_temperature
        )
        semantic_low = self.semantic(features)
        semantic = F.interpolate(semantic_low, size=output_size, mode="bilinear", align_corners=False)
        output={"assignment_low":assignment_low,"embedding":embedding_low,"semantic_logits":semantic}
        if return_full_assignment:
            assignment=F.interpolate(assignment_low,size=output_size,mode="bilinear",align_corners=False); output["assignment"]=assignment/assignment.sum(1,keepdim=True).clamp_min(1e-6)
        if return_tokens:
            # Pool at backbone resolution. Upsampling both tensors before this
            # einsum is mathematically unnecessary.
            mass=assignment_low.sum((2,3)).clamp_min(1e-6); output["region_tokens"]=torch.einsum("bkhw,bdhw->bkd",assignment_low,embedding_low)/mass.unsqueeze(-1)
        return output

    def extract_backbone_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)["out"]

    def forward(self, image: torch.Tensor, return_full_assignment: bool = True, return_tokens: bool = True) -> dict[str, torch.Tensor]:
        features = self.extract_backbone_features(image)
        return self.forward_from_features(
            features, image.shape[-2:], return_full_assignment, return_tokens
        )
