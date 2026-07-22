#!/usr/bin/env python3
"""Train Phase-4 cross-scale variants on frozen multi-scale token cache."""
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
from torch.utils.data import DataLoader, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_4_cross_scale.src.dataset import (
    GroupBalancedSampler,
    TokenCacheDataset,
    collate_token_cache,
)
from benchmarks.v4.phase_4_cross_scale.src.model import CrossScaleModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--label-index", type=Path, required=True)
    p.add_argument("--variant", choices=CrossScaleModel.VARIANTS, required=True)
    p.add_argument("--timestamp")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--samples-per-epoch", type=int)
    p.add_argument("--limit-train-patches", type=int)
    p.add_argument("--limit-val-patches", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--batch-size-per-gpu", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-every-steps", type=int, default=50)
    return p.parse_args()


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def seed_all(value: int):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def all_ranks_true(value, device, world):
    flag = torch.tensor(int(bool(value)), device=device)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def stratified_subset(dataset: TokenCacheDataset, limit: int | None, seed_value: int):
    if limit is None or limit >= len(dataset):
        return dataset
    groups = {}
    for i, row in enumerate(dataset.rows):
        groups.setdefault(row["sampling_group"], []).append(i)
    rng = np.random.default_rng(seed_value)
    total = len(dataset)
    allocation = {name: min(len(ids), int(np.floor(limit * len(ids) / total))) for name, ids in groups.items()}
    remaining = limit - sum(allocation.values())
    order = sorted(groups, key=lambda name: (limit * len(groups[name]) / total - allocation[name], len(groups[name])), reverse=True)
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
            break
    chosen = []
    for name, ids in groups.items():
        chosen.extend(rng.choice(ids, size=allocation[name], replace=False).tolist())
    return Subset(dataset, sorted(chosen))


def metrics_from_logits(logits, labels, ignore):
    keep = labels != ignore
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=ignore)
    correct = ((logits.argmax(-1) == labels) & keep).sum()
    valid = keep.sum()
    return loss, correct, valid


def main():
    ns = parse()
    cfg = load_config(ns.config)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(int(cfg["project"]["seed"]) + rank)

    train_full = TokenCacheDataset(ns.cache_index, ns.label_index, "train")
    val_full = TokenCacheDataset(ns.cache_index, ns.label_index, "val")
    if ns.limit_train_patches:
        train_set: TokenCacheDataset | Subset = Subset(train_full, list(range(ns.limit_train_patches)))
    else:
        train_set = train_full
    val_budget = ns.limit_val_patches if ns.limit_val_patches is not None else int(cfg["training"]["validation_samples"])
    val_selected = stratified_subset(val_full, val_budget, int(cfg["project"]["seed"]))
    val_indices = val_selected.indices if isinstance(val_selected, Subset) else list(range(len(val_selected)))
    if world > 1:
        val_set = Subset(val_full, val_indices[rank::world])
    else:
        val_set = Subset(val_full, val_indices)

    workers = ns.num_workers if ns.num_workers is not None else int(cfg["training"]["num_workers"])
    batch_size = ns.batch_size_per_gpu or int(cfg["training"]["batch_size_per_gpu"])
    loader_kwargs = {"num_workers": workers, "pin_memory": True, "collate_fn": collate_token_cache}
    if workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    budget = ns.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    if isinstance(train_set, Subset):
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    else:
        sampler = GroupBalancedSampler(
            train_set,
            {"class_interior": 0.3, "class_boundary": 0.3, "rare_class": 0.3, "background_or_hard_negative": 0.1},
            int(cfg["project"]["seed"]),
            rank,
            world,
            budget,
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = CrossScaleModel(
        ns.variant,
        dim=int(cfg["model"]["region_dim"]),
        num_classes=int(cfg["data"]["class_count"]),
        num_blocks=int(cfg["model"]["num_blocks"]),
    ).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    stamp = ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["output"]["root"]) / f"phase4_{ns.variant}_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True)
        save(
            output / "run_metadata.json",
            {
                "variant": ns.variant,
                "cache_index": str(ns.cache_index),
                "label_index": str(ns.label_index),
                "world_size": world,
                "batch_size_per_gpu": batch_size,
                "num_workers": workers,
                "samples_per_epoch": budget,
                "validation_samples": val_budget,
                "config": cfg,
            },
        )
    if world > 1:
        dist.barrier()

    start_epoch, history = 0, []
    if ns.resume:
        ckpt = torch.load(ns.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, history = int(ckpt["epoch"]) + 1, list(ckpt.get("history", []))

    epochs = ns.max_epochs or int(cfg["training"]["epochs"])
    ignore = int(cfg["data"]["ignore_index"])
    amp = torch.bfloat16 if cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    if rank == 0:
        print(
            {
                "event": "train_start",
                "variant": ns.variant,
                "world_size": world,
                "train_steps_per_rank": len(train_loader),
                "val_patches_per_rank": len(val_set),
                "epochs": epochs,
                "start_epoch": start_epoch,
            },
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        train_loss = train_correct = train_valid = 0.0
        steps = 0
        t0 = time.monotonic()
        for batch in train_loader:
            for key in (
                "fine_tokens",
                "middle_tokens",
                "coarse_tokens",
                "fine_middle_edge_index",
                "fine_middle_edge_weight",
                "middle_coarse_edge_index",
                "middle_coarse_edge_weight",
                "labels",
            ):
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp):
                logits = model(batch)
                loss, correct, valid = metrics_from_logits(logits, batch["labels"], ignore)
            if not all_ranks_true(torch.isfinite(loss).item(), device, world):
                raise FloatingPointError(f"non-finite loss epoch={epoch} step={steps} rank={rank}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
            if not all_ranks_true(torch.isfinite(grad).item(), device, world):
                raise FloatingPointError(f"non-finite grad epoch={epoch} step={steps} rank={rank}")
            optimizer.step()
            train_loss += float(loss.detach())
            train_correct += float(correct.detach())
            train_valid += float(valid.detach())
            steps += 1
            if rank == 0 and ns.log_every_steps > 0 and steps % ns.log_every_steps == 0:
                print(
                    {
                        "event": "train_progress",
                        "variant": ns.variant,
                        "epoch": epoch,
                        "step": steps,
                        "loss_running": train_loss / steps,
                        "elapsed_sec": round(time.monotonic() - t0, 1),
                    },
                    flush=True,
                )

        model.eval()
        val_loss = val_correct = val_valid = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                for key in (
                    "fine_tokens",
                    "middle_tokens",
                    "coarse_tokens",
                    "fine_middle_edge_index",
                    "fine_middle_edge_weight",
                    "middle_coarse_edge_index",
                    "middle_coarse_edge_weight",
                    "labels",
                ):
                    batch[key] = batch[key].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp):
                    logits = model(batch)
                    loss, correct, valid = metrics_from_logits(logits, batch["labels"], ignore)
                val_loss += float(loss)
                val_correct += float(correct)
                val_valid += float(valid)
                val_steps += 1

        values = torch.tensor(
            [train_loss, steps, train_correct, train_valid, val_loss, val_steps, val_correct, val_valid],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(values)
        if rank == 0:
            train_loss, steps, train_correct, train_valid, val_loss, val_steps, val_correct, val_valid = values.tolist()
            row = {
                "epoch": epoch,
                "variant": ns.variant,
                "train_loss": train_loss / max(steps, 1),
                "train_region_acc": train_correct / max(train_valid, 1),
                "val_loss": val_loss / max(val_steps, 1),
                "val_region_acc": val_correct / max(val_valid, 1),
                "epoch_elapsed_sec": round(time.monotonic() - t0, 1),
            }
            history.append(row)
            ckpt_path = output / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "variant": ns.variant,
                    "model": (model.module if world > 1 else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "config": cfg,
                },
                ckpt_path,
            )
            save(output / "last_checkpoint.json", {"epoch": epoch, "path": str(ckpt_path)})
            best = max(history, key=lambda x: x["val_region_acc"])
            save(
                output / "best_checkpoint.json",
                {
                    "epoch": best["epoch"],
                    "path": str(output / f"checkpoint_epoch_{best['epoch']:03d}.pth"),
                    "val_region_acc": best["val_region_acc"],
                },
            )
            save(output / "metrics.json", history)
            print(row, flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
