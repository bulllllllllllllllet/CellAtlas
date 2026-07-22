#!/usr/bin/env python3
"""Evaluate full-cell, zero-cell, and independently trained region-only controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells
from benchmarks.v4.phase_3_cell_region.train_phase3 import all_ranks_true, region_labels, save, stratified_subset


VARIANTS = ("cell_full", "cell_zeroed", "region_only")


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
    parser.add_argument("--log-every-steps", type=int, default=100)
    return parser.parse_args()


def checkpoint_models(checkpoint_path, cfg, device, require_cell):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    has_cell = checkpoint.get("cell") is not None
    if has_cell != require_cell:
        expected = "cell" if require_cell else "region_only"
        raise ValueError(f"{checkpoint_path} is not a {expected} checkpoint")
    head = torch.nn.Linear(int(cfg["model"]["region_dim"]), int(cfg["data"]["class_count"])).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    cell = None
    if require_cell:
        cell = CellToRegionAttention(int(cfg["model"]["region_dim"]), int(cfg["data"]["cell_feature_dim"])).to(device)
        cell.load_state_dict(checkpoint["cell"], strict=True)
        cell.eval()
    return cell, head, int(checkpoint["epoch"])


def summarize(confusion, loss_sum, valid_count, class_names):
    matrix = confusion.astype(np.float64)
    true_positive = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)
    present = support > 0
    rows = [{"class_id": i, "class_name": class_names[i], "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(class_names))]
    summary = {
        "cross_entropy": float(loss_sum / max(valid_count, 1)),
        "region_accuracy": float(true_positive.sum() / max(valid_count, 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(valid_count, 1)),
        "valid_regions": int(valid_count),
    }
    return summary, rows


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
    output = args.output_root / f"phase3_cell_ablation_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)

    dataset = CellRegionDataset(args.patch_index, args.slic_index, args.val_routing, p2cfg, "val")
    validation_samples = args.validation_samples or int(cfg["training"]["validation_samples"])
    selected = stratified_subset(dataset, validation_samples, int(cfg["project"]["seed"]))
    selected_indices = selected.indices if isinstance(selected, Subset) else list(range(len(selected)))
    selected_hash = hashlib.sha256(np.asarray(selected_indices, dtype=np.int64).tobytes()).hexdigest()
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
    cell, cell_head, cell_epoch = checkpoint_models(args.cell_checkpoint, cfg, device, require_cell=True)
    _, region_head, region_epoch = checkpoint_models(args.region_only_checkpoint, cfg, device, require_cell=False)

    classes = int(cfg["data"]["class_count"])
    ignore = int(cfg["data"]["ignore_index"])
    amp = torch.bfloat16 if cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    confusion = {name: torch.zeros((classes, classes), device=device, dtype=torch.int64) for name in VARIANTS}
    loss_sum = {name: torch.zeros((), device=device, dtype=torch.float64) for name in VARIANTS}
    valid_count = torch.zeros((), device=device, dtype=torch.int64)

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
                zero_tokens = cell(
                    phase2_output["region_tokens"], mass, cells,
                    torch.zeros_like(cell_valid), torch.zeros_like(total_count),
                )["fused_tokens"]
                logits = {
                    "cell_full": cell_head(full_tokens),
                    "cell_zeroed": cell_head(zero_tokens),
                    "region_only": region_head(phase2_output["region_tokens"]),
                }
            finite_logits = all(torch.isfinite(value).all().item() for value in logits.values())
            if not all_ranks_true(finite_logits, device, world):
                raise FloatingPointError(f"non-finite ablation logits rank={rank} step={step} amp={amp}")
            keep = labels != ignore
            local_valid = keep.sum()
            valid_count += local_valid
            for name, prediction_logits in logits.items():
                loss_sum[name] += F.cross_entropy(
                    prediction_logits.flatten(0, 1), labels.flatten(), ignore_index=ignore, reduction="sum"
                ).double()
                prediction = prediction_logits.argmax(-1)
                keys = labels[keep] * classes + prediction[keep]
                confusion[name] += torch.bincount(keys, minlength=classes * classes).reshape(classes, classes)
            if rank == 0 and args.log_every_steps > 0 and step % args.log_every_steps == 0:
                print({"event": "evaluation_progress", "step": step, "steps_per_rank": len(loader)}, flush=True)

    for name in VARIANTS:
        if world > 1:
            dist.all_reduce(confusion[name])
            dist.all_reduce(loss_sum[name])
    if world > 1:
        dist.all_reduce(valid_count)

    if rank == 0:
        output.mkdir(parents=True)
        class_names = [row["name"] for row in p2cfg["data"]["class_map"]]
        summaries = {}
        for name in VARIANTS:
            matrix = confusion[name].cpu().numpy()
            summary, rows = summarize(matrix, float(loss_sum[name]), int(valid_count), class_names)
            summaries[name] = summary
            np.save(output / f"confusion_matrix_{name}.npy", matrix)
            with (output / f"per_class_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("class_id", "class_name", "precision", "recall", "f1", "support"))
                writer.writeheader()
                writer.writerows(rows)
        comparison = {
            "cell_full_minus_region_only_accuracy": summaries["cell_full"]["region_accuracy"] - summaries["region_only"]["region_accuracy"],
            "cell_full_minus_zeroed_accuracy": summaries["cell_full"]["region_accuracy"] - summaries["cell_zeroed"]["region_accuracy"],
            "cell_full_minus_region_only_macro_f1": summaries["cell_full"]["macro_f1"] - summaries["region_only"]["macro_f1"],
            "cell_full_minus_zeroed_macro_f1": summaries["cell_full"]["macro_f1"] - summaries["cell_zeroed"]["macro_f1"],
        }
        save(output / "summary.json", {"variants": summaries, "comparison": comparison})
        save(output / "run_metadata.json", {
            "validation_samples": validation_samples,
            "selected_index_sha256": selected_hash,
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
        })
        print({"event": "evaluation_complete", "output": str(output), **comparison}, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
