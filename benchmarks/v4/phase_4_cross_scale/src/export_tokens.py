"""Shared frozen multi-scale token export path used by singleton and cache jobs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells
from benchmarks.v4.phase_4_cross_scale.src.geometry import (
    assignment_centroids_areas,
    parent_child_edges,
    scale_level0_box,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def he_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div_(255.0)
    mean = torch.from_numpy(IMAGENET_MEAN)[:, None, None]
    std = torch.from_numpy(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def load_phase2(phase2_config: dict, checkpoint: Path, device: torch.device) -> DeepRegionEncoder:
    model = DeepRegionEncoder(
        phase2_config,
        len(phase2_config["data"]["class_map"]),
        int(phase2_config["model"]["num_regions"]),
        int(phase2_config["model"]["embedding_dim"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"phase2 load mismatch missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def load_cell_module(region_dim: int, cell_dim: int, checkpoint: Path, device: torch.device) -> CellToRegionAttention:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("cell") is None:
        raise ValueError(f"{checkpoint} is not a cell checkpoint")
    cell = CellToRegionAttention(region_dim, cell_dim).to(device)
    cell.load_state_dict(ckpt["cell"], strict=True)
    cell.eval()
    return cell


@torch.no_grad()
def encode_scale(
    model: DeepRegionEncoder,
    image_rgb: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray | torch.Tensor]:
    tensor = he_to_tensor(image_rgb).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        out = model(tensor, return_full_assignment=True, return_tokens=True)
    assignment = out["assignment"][0].float().cpu().numpy()
    assignment_low = out["assignment_low"][0].float().cpu().numpy()
    tokens = out["region_tokens"][0].float().cpu().numpy()
    return {
        "assignment": assignment,
        "assignment_low": assignment_low,
        "tokens": tokens,
        "image_rgb": image_rgb,
    }


@torch.no_grad()
def encode_scale_batch(
    model: DeepRegionEncoder,
    images: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    return_full_assignment: bool = False,
) -> dict[str, np.ndarray]:
    """Batched Phase-2 encode. ``images`` is already normalized ``[B,3,H,W]``."""
    images = images.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        out = model(images, return_full_assignment=return_full_assignment, return_tokens=True)
    result = {
        "assignment_low": out["assignment_low"].float().cpu().numpy(),
        "tokens": out["region_tokens"].float().cpu().numpy(),
    }
    if return_full_assignment:
        result["assignment"] = out["assignment"].float().cpu().numpy()
    return result


@torch.no_grad()
def fuse_fine_cells(
    phase2: DeepRegionEncoder,
    cell: CellToRegionAttention,
    image_rgb: np.ndarray,
    cells: torch.Tensor,
    cell_valid: torch.Tensor,
    total_cell_count: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    tensor = he_to_tensor(image_rgb).unsqueeze(0).to(device)
    cells = cells.to(device)
    cell_valid = cell_valid.to(device)
    total_cell_count = total_cell_count.to(device)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        out = phase2(tensor, return_full_assignment=True, return_tokens=True)
        mass = sample_assignment_at_cells(out["assignment_low"], cells)
        fused = cell(out["region_tokens"], mass, cells, cell_valid, total_cell_count)["fused_tokens"]
    return {
        "assignment": out["assignment"][0].float().cpu().numpy(),
        "assignment_low": out["assignment_low"][0].float().cpu().numpy(),
        "tokens": out["region_tokens"][0].float().cpu().numpy(),
        "fused_tokens": fused[0].float().cpu().numpy(),
        "image_rgb": image_rgb,
    }


@torch.no_grad()
def fuse_fine_cells_batch(
    phase2: DeepRegionEncoder,
    cell: CellToRegionAttention,
    images: torch.Tensor,
    cells: torch.Tensor,
    cell_valid: torch.Tensor,
    total_cell_count: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    return_full_assignment: bool = False,
) -> dict[str, np.ndarray]:
    images = images.to(device, non_blocking=True)
    cells = cells.to(device, non_blocking=True)
    cell_valid = cell_valid.to(device, non_blocking=True)
    total_cell_count = total_cell_count.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        out = phase2(images, return_full_assignment=return_full_assignment, return_tokens=True)
        mass = sample_assignment_at_cells(out["assignment_low"], cells)
        fused = cell(out["region_tokens"], mass, cells, cell_valid, total_cell_count)["fused_tokens"]
    result = {
        "assignment_low": out["assignment_low"].float().cpu().numpy(),
        "tokens": out["region_tokens"].float().cpu().numpy(),
        "fused_tokens": fused.float().cpu().numpy(),
    }
    if return_full_assignment:
        result["assignment"] = out["assignment"].float().cpu().numpy()
    return result


def read_scale_image(row: dict, scale: str, output_size: int = 512) -> np.ndarray:
    x, y, w, h = scale_level0_box(row, scale)
    return read_he_patch(Path(row["wsi_path"]), x, y, w, h, output_size)


def export_patch_multiscale(
    row: dict,
    phase2: DeepRegionEncoder,
    cell: CellToRegionAttention | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    cells: torch.Tensor | None = None,
    cell_valid: torch.Tensor | None = None,
    total_cell_count: torch.Tensor | None = None,
    top_k: int = 4,
) -> dict:
    """Export three-scale tokens, physical centroids, and sparse parent-child edges."""
    scales = ("10x", "5x", "2p5x")
    packed = {}
    for scale in scales:
        image = read_scale_image(row, scale)
        box = scale_level0_box(row, scale)
        if scale == "10x" and cell is not None and cells is not None:
            encoded = fuse_fine_cells(
                phase2, cell, image, cells, cell_valid, total_cell_count, device, amp_dtype
            )
            tokens = encoded["fused_tokens"]
            region_tokens = encoded["tokens"]
        else:
            encoded = encode_scale(phase2, image, device, amp_dtype)
            tokens = encoded["tokens"]
            region_tokens = encoded["tokens"]
        geom = assignment_centroids_areas(encoded["assignment"], *box)
        packed[scale] = {
            "image_rgb": encoded["image_rgb"],
            "assignment": encoded["assignment"],
            "assignment_low": encoded["assignment_low"],
            "tokens": tokens.astype(np.float32),
            "region_tokens": region_tokens.astype(np.float32),
            "box_level0": box,
            **geom,
        }
    fine_middle = parent_child_edges(
        packed["10x"]["assignment"],
        packed["10x"]["box_level0"],
        packed["5x"]["assignment"],
        packed["5x"]["box_level0"],
        top_k=top_k,
    )
    middle_coarse = parent_child_edges(
        packed["5x"]["assignment"],
        packed["5x"]["box_level0"],
        packed["2p5x"]["assignment"],
        packed["2p5x"]["box_level0"],
        top_k=top_k,
    )
    return {
        "patch_id": row["patch_id"],
        "wsi_id": row["wsi_id"],
        "scales": packed,
        "edges": {
            "fine_middle": fine_middle,
            "middle_coarse": middle_coarse,
        },
    }
