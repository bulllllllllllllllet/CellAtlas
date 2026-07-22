#!/usr/bin/env python3
"""Export Phase-3 region embeddings on the fixed validation subset (read-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
# RegionDataset/libvips must import before DeepRegionEncoder/torchvision.
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.evaluate_phase3_ablation import checkpoint_models
from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_3_cell_region.train_phase3 import all_ranks_true, region_labels, save, stratified_subset


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--slic-index", type=Path, required=True)
    parser.add_argument("--val-routing", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--region-only-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase3/evaluation"))
    parser.add_argument("--timestamp")
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--expected-selected-hash")
    return parser.parse_args()


def main():
    args = parse()
    cfg, p2cfg = load_config(args.config), load_config(args.phase2_config)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"phase3_embedding_closeout_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)

    dataset = CellRegionDataset(args.patch_index, args.slic_index, args.val_routing, p2cfg, "val")
    validation_samples = args.validation_samples or int(cfg["training"]["validation_samples"])
    selected = stratified_subset(dataset, validation_samples, int(cfg["project"]["seed"]))
    selected_indices = selected.indices if isinstance(selected, Subset) else list(range(len(selected)))
    selected_hash = hashlib.sha256(np.asarray(selected_indices, dtype=np.int64).tobytes()).hexdigest()
    if args.expected_selected_hash and selected_hash != args.expected_selected_hash:
        raise RuntimeError(
            f"validation subset hash mismatch: got {selected_hash}, expected {args.expected_selected_hash}"
        )
    if world > 1:
        selected = Subset(selected, list(range(rank, len(selected), world)))
    workers = args.num_workers if args.num_workers is not None else int(cfg["training"]["num_workers"])
    batch_size = args.batch_size_per_gpu or int(cfg["training"]["batch_size_per_gpu"])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": lambda items: collate_cell_region(items, int(cfg["data"]["max_cells_per_patch"])),
    }
    if workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=1)
    loader = DataLoader(selected, **loader_kwargs)

    phase2 = DeepRegionEncoder(
        p2cfg, len(p2cfg["data"]["class_map"]), int(cfg["model"]["num_regions"]), int(cfg["model"]["region_dim"])
    ).to(device)
    phase2_state = torch.load(cfg["model"]["phase2_checkpoint"], map_location=device, weights_only=False)["model"]
    missing, unexpected = phase2.load_state_dict(phase2_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"phase2 checkpoint mismatch missing={len(missing)} unexpected={len(unexpected)}")
    phase2.eval()
    cell, _, cell_epoch = checkpoint_models(args.cell_checkpoint, cfg, device, require_cell=True)
    _, _, region_epoch = checkpoint_models(args.region_only_checkpoint, cfg, device, require_cell=False)

    classes = int(cfg["data"]["class_count"])
    ignore = int(cfg["data"]["ignore_index"])
    amp = torch.bfloat16 if cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    local = {
        "cell_full": [],
        "region_only": [],
        "labels": [],
        "patch_ids": [],
        "region_ids": [],
        "wsi_ids": [],
    }

    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            image, masks = batch["image"].to(device), batch["mask"].to(device)
            cells = batch["cells"].to(device)
            cell_valid = batch["cell_valid"].to(device)
            total_count = batch["total_cell_count"].to(device)
            with torch.autocast("cuda", dtype=amp):
                phase2_output = phase2(image, return_full_assignment=False, return_tokens=True)
                labels = region_labels(phase2_output["assignment_low"], masks, classes, ignore)
                mass = sample_assignment_at_cells(phase2_output["assignment_low"], cells)
                full_tokens = cell(phase2_output["region_tokens"], mass, cells, cell_valid, total_count)["fused_tokens"]
                region_tokens = phase2_output["region_tokens"]
            if not all_ranks_true(
                torch.isfinite(full_tokens).all().item() and torch.isfinite(region_tokens).all().item(),
                device,
                world,
            ):
                raise FloatingPointError(f"non-finite embeddings rank={rank} step={step}")
            keep = labels != ignore
            if keep.any():
                b_idx, k_idx = torch.where(keep)
                local["cell_full"].append(full_tokens[b_idx, k_idx].float().cpu().numpy())
                local["region_only"].append(region_tokens[b_idx, k_idx].float().cpu().numpy())
                local["labels"].append(labels[b_idx, k_idx].cpu().numpy().astype(np.int16))
                for b, k in zip(b_idx.tolist(), k_idx.tolist(), strict=True):
                    local["patch_ids"].append(batch["patch_id"][b])
                    local["region_ids"].append(int(k))
                    local["wsi_ids"].append(batch["patch_id"][b].split("__", 1)[0])
            if rank == 0 and args.log_every_steps > 0 and step % args.log_every_steps == 0:
                print({"event": "export_progress", "step": step, "steps_per_rank": len(loader)}, flush=True)

    labels_arr = np.concatenate(local["labels"], axis=0) if local["labels"] else np.zeros((0,), np.int16)
    payload = {
        "cell_full": np.concatenate(local["cell_full"], axis=0) if local["cell_full"] else np.zeros((0, int(cfg["model"]["region_dim"])), np.float32),
        "region_only": np.concatenate(local["region_only"], axis=0) if local["region_only"] else np.zeros((0, int(cfg["model"]["region_dim"])), np.float32),
        "labels": labels_arr,
        "patch_ids": np.asarray(local["patch_ids"], dtype=object),
        "region_ids": np.asarray(local["region_ids"], dtype=np.int16),
        "wsi_ids": np.asarray(local["wsi_ids"], dtype=object),
        "rank": np.full(len(labels_arr), rank, dtype=np.int16),
    }
    if world > 1:
        gathered = [None for _ in range(world)] if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]

    if rank == 0:
        output.mkdir(parents=True)
        parts = [item for item in gathered if item is not None and len(item["labels"])]
        if not parts:
            raise RuntimeError("no valid region embeddings exported")
        merged = {
            "cell_full": np.concatenate([p["cell_full"] for p in parts], axis=0).astype(np.float16),
            "region_only": np.concatenate([p["region_only"] for p in parts], axis=0).astype(np.float16),
            "labels": np.concatenate([p["labels"] for p in parts], axis=0),
            "patch_ids": np.concatenate([p["patch_ids"] for p in parts], axis=0),
            "region_ids": np.concatenate([p["region_ids"] for p in parts], axis=0),
            "wsi_ids": np.concatenate([p["wsi_ids"] for p in parts], axis=0),
            "rank": np.concatenate([p["rank"] for p in parts], axis=0),
        }
        for key in ("cell_full", "region_only", "labels", "region_ids", "rank"):
            np.save(output / f"{key}.npy", merged[key])
        np.save(output / "patch_ids.npy", merged["patch_ids"], allow_pickle=True)
        np.save(output / "wsi_ids.npy", merged["wsi_ids"], allow_pickle=True)
        save(
            output / "export_metadata.json",
            {
                "validation_samples": validation_samples,
                "selected_index_sha256": selected_hash,
                "valid_regions": int(len(merged["labels"])),
                "world_size": world,
                "batch_size_per_gpu": batch_size,
                "num_workers_per_rank": workers,
                "cell_checkpoint": str(args.cell_checkpoint),
                "cell_checkpoint_epoch": cell_epoch,
                "region_only_checkpoint": str(args.region_only_checkpoint),
                "region_only_checkpoint_epoch": region_epoch,
                "phase2_checkpoint": cfg["model"]["phase2_checkpoint"],
                "phase2_missing": len(missing),
                "phase2_unexpected": len(unexpected),
                "split": "val",
                "embedding_dtype": "float16",
                "variants": ["cell_full", "region_only"],
            },
        )
        print(
            {
                "event": "export_complete",
                "output": str(output),
                "valid_regions": int(len(merged["labels"])),
                "selected_index_sha256": selected_hash,
            },
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
