#!/usr/bin/env python3
"""Train a frozen Phase-2 region probe, with or without cell-to-region injection."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.dataset import BalancedPatchSampler
from benchmarks.v4.phase_2_region_encoder.src.dataset import RegionDataset
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.src.dataset import CellRegionDataset, collate_cell_region
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase2-config", type=Path, required=True)
    p.add_argument("--patch-index", type=Path, required=True)
    p.add_argument("--slic-index", type=Path, required=True)
    p.add_argument("--train-routing", type=Path)
    p.add_argument("--val-routing", type=Path)
    p.add_argument("--variant", choices=("cell", "region_only"), default="cell")
    p.add_argument("--timestamp")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--samples-per-epoch", type=int)
    p.add_argument("--limit-train-patches", type=int)
    p.add_argument("--limit-val-patches", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-every-steps", type=int, default=100)
    return p.parse_args()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def all_ranks_true(value, device, world):
    flag = torch.tensor(int(bool(value)), device=device)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def dataset_rows(dataset):
    return dataset.region.rows if hasattr(dataset, "region") else dataset.rows


def stratified_subset(dataset, limit, seed_value):
    """Fixed proportional validation subset over manifest sampling groups."""
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("validation sample limit must be positive")
    groups = {}
    for i, row in enumerate(dataset_rows(dataset)):
        groups.setdefault(row["sampling_group"], []).append(i)
    rng = np.random.default_rng(seed_value)
    total = len(dataset)
    allocation = {
        name: min(len(ids), int(np.floor(limit * len(ids) / total)))
        for name, ids in groups.items()
    }
    remaining = limit - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (limit * len(groups[name]) / total - allocation[name], len(groups[name])),
        reverse=True,
    )
    while remaining:
        progressed = False
        for name in order:
            if allocation[name] < len(groups[name]):
                allocation[name] += 1
                remaining -= 1
                progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("could not allocate validation subset")
    chosen = []
    for name, ids in groups.items():
        chosen.extend(rng.choice(ids, size=allocation[name], replace=False).tolist())
    return Subset(dataset, sorted(chosen))


def region_labels(assignment, mask, classes, ignore):
    hard = F.interpolate(
        assignment, size=mask.shape[-2:], mode="bilinear", align_corners=False
    ).argmax(1)
    output = torch.full(
        (mask.shape[0], assignment.shape[1]), ignore, device=mask.device, dtype=torch.long
    )
    for b in range(mask.shape[0]):
        valid = mask[b] != ignore
        if valid.any():
            votes = torch.bincount(
                hard[b][valid] * classes + mask[b][valid],
                minlength=assignment.shape[1] * classes,
            ).reshape(assignment.shape[1], classes)
            present = votes.sum(1) > 0
            output[b, present] = votes[present].argmax(1)
    return output


def collate_region(items):
    return {
        "image": torch.stack([x["image"] for x in items]),
        "mask": torch.stack([x["mask"] for x in items]),
        "slic": torch.stack([x["slic"] for x in items]),
        "patch_id": [x["patch_id"] for x in items],
    }


def representation(variant, cell, phase2_output, batch, device):
    if variant == "region_only":
        return phase2_output["region_tokens"]
    cells = batch["cells"].to(device)
    valid_cells = batch["cell_valid"].to(device)
    true_count = batch["total_cell_count"].to(device)
    mass = sample_assignment_at_cells(phase2_output["assignment_low"], cells)
    return cell(
        phase2_output["region_tokens"], mass, cells, valid_cells, true_count
    )["fused_tokens"]


def main():
    ns = parse()
    cfg = load_config(ns.config)
    p2cfg = load_config(ns.phase2_config)
    if ns.variant == "cell" and (ns.train_routing is None or ns.val_routing is None):
        raise ValueError("--train-routing and --val-routing are required for variant=cell")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed(int(cfg["project"]["seed"]) + rank)

    if ns.variant == "cell":
        train_set = CellRegionDataset(ns.patch_index, ns.slic_index, ns.train_routing, p2cfg, "train")
        val_set = CellRegionDataset(ns.patch_index, ns.slic_index, ns.val_routing, p2cfg, "val")
        collate = lambda items: collate_cell_region(items, int(cfg["data"]["max_cells_per_patch"]))
    else:
        train_set = RegionDataset(ns.patch_index, ns.slic_index, p2cfg, "train", augment=False)
        val_set = RegionDataset(ns.patch_index, ns.slic_index, p2cfg, "val", augment=False)
        collate = collate_region
    if ns.limit_train_patches:
        train_set = Subset(train_set, list(range(ns.limit_train_patches)))
    val_budget = ns.limit_val_patches if ns.limit_val_patches is not None else int(cfg["training"]["validation_samples"])
    val_set = stratified_subset(val_set, val_budget, int(cfg["project"]["seed"]))

    workers = ns.num_workers if ns.num_workers is not None else int(cfg["training"]["num_workers"])
    loader_kwargs = {"num_workers": workers, "pin_memory": True, "collate_fn": collate}
    if workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=1)
    budget = ns.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    if isinstance(train_set, Subset):
        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    else:
        sampler = BalancedPatchSampler(
            train_set.region if hasattr(train_set, "region") else train_set,
            {"class_interior": 0.3, "class_boundary": 0.3, "rare_class": 0.3, "background_or_hard_negative": 0.1},
            int(cfg["project"]["seed"]), rank, world, budget,
        )
    if world > 1:
        val_set = Subset(val_set, list(range(rank, len(val_set), world)))
    batch_size = int(cfg["training"]["batch_size_per_gpu"])
    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    phase2 = DeepRegionEncoder(
        p2cfg, len(p2cfg["data"]["class_map"]), int(cfg["model"]["num_regions"]), int(cfg["model"]["region_dim"])
    ).to(device)
    state = torch.load(cfg["model"]["phase2_checkpoint"], map_location=device, weights_only=False)["model"]
    missing, unexpected = phase2.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"phase2 checkpoint mismatch missing={len(missing)} unexpected={len(unexpected)}")
    phase2.eval()
    for parameter in phase2.parameters():
        parameter.requires_grad = False

    cell = (
        CellToRegionAttention(int(cfg["model"]["region_dim"]), int(cfg["data"]["cell_feature_dim"])).to(device)
        if ns.variant == "cell" else None
    )
    head = torch.nn.Linear(int(cfg["model"]["region_dim"]), int(cfg["data"]["class_count"])).to(device)
    if world > 1:
        if cell is not None:
            cell = torch.nn.parallel.DistributedDataParallel(cell, device_ids=[local_rank])
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[local_rank])
    parameters = list(head.parameters()) + (list(cell.parameters()) if cell is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    stamp = ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "phase_3_cell_region_train" if ns.variant == "cell" else "phase_3_region_only_train"
    output = Path(cfg["output"]["root"]) / f"{prefix}_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True)
        save(output / "config_snapshot.json", {"phase3": cfg, "phase2": p2cfg, "variant": ns.variant})
        save(output / "run_metadata.json", {
            "variant": ns.variant,
            "train_routing": str(ns.train_routing) if ns.train_routing else None,
            "val_routing": str(ns.val_routing) if ns.val_routing else None,
            "phase2_checkpoint": cfg["model"]["phase2_checkpoint"],
            "frozen_phase2": True,
            "world_size": world,
            "checkpoint_missing": len(missing),
            "checkpoint_unexpected": len(unexpected),
        })
    if world > 1:
        dist.barrier()

    start_epoch, history = 0, []
    if ns.resume:
        checkpoint = torch.load(ns.resume, map_location=device, weights_only=False)
        if (checkpoint["cell"] is None) != (cell is None):
            raise ValueError(f"resume checkpoint is incompatible with variant={ns.variant}")
        if cell is not None:
            (cell.module if world > 1 else cell).load_state_dict(checkpoint["cell"])
        (head.module if world > 1 else head).load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, history = checkpoint["epoch"] + 1, checkpoint["history"]

    epochs = ns.max_epochs or int(cfg["training"]["epochs"])
    classes = int(cfg["data"]["class_count"])
    ignore = int(cfg["data"]["ignore_index"])
    amp = torch.bfloat16 if cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    if rank == 0:
        print({"event": "train_start", "variant": ns.variant, "world_size": world,
               "train_samples_per_rank": len(sampler) if sampler else len(train_set),
               "validation_samples": val_budget, "validation_samples_per_rank": len(val_set),
               "batch_size_per_gpu": batch_size, "epochs": epochs, "start_epoch": start_epoch}, flush=True)

    for epoch in range(start_epoch, epochs):
        if sampler:
            sampler.set_epoch(epoch)
        if cell is not None:
            cell.train()
        head.train()
        total = valid = correct = 0.0
        steps = 0
        epoch_start = time.monotonic()
        for batch in loader:
            image, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                phase2_output = phase2(image, return_full_assignment=False, return_tokens=True)
            labels = region_labels(phase2_output["assignment_low"], masks, classes, ignore)
            with torch.autocast("cuda", dtype=amp):
                logits = head(representation(ns.variant, cell, phase2_output, batch, device))
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=ignore)
            if not all_ranks_true(torch.isfinite(loss).item(), device, world):
                raise FloatingPointError(f"non-finite loss epoch={epoch} step={steps} rank={rank} amp={amp}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, float(cfg["training"]["max_grad_norm"]))
            if not all_ranks_true(torch.isfinite(grad).item(), device, world):
                raise FloatingPointError(f"non-finite grad epoch={epoch} step={steps} rank={rank} amp={amp}")
            optimizer.step()
            keep = labels != ignore
            total += float(loss.detach())
            valid += keep.sum().item()
            correct += ((logits.argmax(-1) == labels) & keep).sum().item()
            steps += 1
            if rank == 0 and ns.log_every_steps > 0 and steps % ns.log_every_steps == 0:
                print({"event": "train_progress", "variant": ns.variant, "epoch": epoch,
                       "step": steps, "steps_per_rank": len(loader), "loss_running": total / steps,
                       "elapsed_sec": round(time.monotonic() - epoch_start, 1)}, flush=True)

        if cell is not None:
            cell.eval()
        head.eval()
        val_total = val_valid = val_correct = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image, masks = batch["image"].to(device), batch["mask"].to(device)
                with torch.autocast("cuda", dtype=amp):
                    phase2_output = phase2(image, return_full_assignment=False, return_tokens=True)
                    labels = region_labels(phase2_output["assignment_low"], masks, classes, ignore)
                    logits = head(representation(ns.variant, cell, phase2_output, batch, device))
                    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=ignore)
                keep = labels != ignore
                val_total += float(loss)
                val_valid += keep.sum().item()
                val_correct += ((logits.argmax(-1) == labels) & keep).sum().item()
        values = torch.tensor([total, steps, correct, valid, val_total, len(val_loader), val_correct, val_valid], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(values)
        if rank == 0:
            total, steps, correct, valid, val_total, val_steps, val_correct, val_valid = values.tolist()
            row = {"epoch": epoch, "train_loss": total / max(steps, 1), "train_region_acc": correct / max(valid, 1),
                   "val_loss": val_total / max(val_steps, 1), "val_region_acc": val_correct / max(val_valid, 1),
                   "epoch_elapsed_sec": round(time.monotonic() - epoch_start, 1)}
            history.append(row)
            checkpoint_path = output / f"checkpoint_epoch_{epoch:03d}.pth"
            cell_state = (cell.module if world > 1 else cell).state_dict() if cell is not None else None
            torch.save({"epoch": epoch, "variant": ns.variant, "cell": cell_state,
                        "head": (head.module if world > 1 else head).state_dict(),
                        "optimizer": optimizer.state_dict(), "history": history,
                        "config": {"phase3": cfg, "phase2": p2cfg}}, checkpoint_path)
            save(output / "last_checkpoint.json", {"epoch": epoch, "path": str(checkpoint_path)})
            best = max(history, key=lambda value: value["val_region_acc"])
            save(output / "best_checkpoint.json", {"epoch": best["epoch"],
                 "path": str(output / f"checkpoint_epoch_{best['epoch']:03d}.pth"),
                 "val_region_acc": best["val_region_acc"]})
            save(output / "metrics.json", history)
            print(row, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
