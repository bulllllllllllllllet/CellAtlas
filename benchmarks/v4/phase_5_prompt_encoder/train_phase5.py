#!/usr/bin/env python3
"""Train the Phase-5 prompt set encoder on frozen fine-region tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import EpisodeBalancedSampler, PromptEpisodeDataset, collate_prompt_episodes
from benchmarks.v4.phase_5_prompt_encoder.src.losses import metric_counts, metrics_from_counts, prompt_region_loss
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel, load_fine_norm_from_phase4


TENSOR_KEYS = (
    "prompt_size_id", "target_class", "fine_tokens", "fine_active", "region_xy", "region_area",
    "positive_tokens", "negative_tokens", "positive_xy", "negative_xy", "positive_mask",
    "negative_mask", "prompted_regions", "binary_target",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--label-index", type=Path, required=True)
    p.add_argument("--patch-index", type=Path, required=True)
    p.add_argument("--eligibility-index", type=Path, required=True)
    p.add_argument("--phase4-checkpoint", type=Path, required=True)
    p.add_argument("--timestamp")
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--stop-after-epoch", type=int, help="Inclusive early-stop epoch without changing scheduler horizon")
    p.add_argument("--samples-per-epoch", type=int)
    p.add_argument("--validation-samples", type=int)
    p.add_argument("--limit-train-episodes", type=int)
    p.add_argument("--limit-val-episodes", type=int)
    p.add_argument("--batch-size-per-gpu", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--log-every-steps", type=int, default=50)
    return p.parse_args()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path("benchmarks/v4/phase_5_prompt_encoder/src/dataset.py"),
        Path("benchmarks/v4/phase_5_prompt_encoder/src/model.py"),
        Path("benchmarks/v4/phase_5_prompt_encoder/src/losses.py"),
        Path("benchmarks/v4/phase_5_prompt_encoder/src/prompts.py"),
        Path("benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def all_ranks_true(value: bool, device: torch.device, world: int) -> bool:
    flag = torch.tensor(int(value), device=device)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def move_batch(batch: dict, device: torch.device) -> dict:
    for key in TENSOR_KEYS:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def finite_outputs(output: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in output.values())


def make_dataset(cfg, ns, split):
    return PromptEpisodeDataset(
        ns.cache_index, ns.label_index, ns.patch_index, split,
        seed=int(cfg["project"]["seed"]),
        size_probabilities=cfg["data"]["size_probabilities"],
        target_class_ids=tuple(cfg["data"]["class_ids"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
        centroid_knn=int(cfg["data"]["centroid_knn"]),
        eligibility_index=ns.eligibility_index,
    )


def main() -> None:
    ns = parse_args(); cfg = load_config(ns.config)
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-5 training requires CUDA")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    seed_all(int(cfg["project"]["seed"]) + rank)

    train_full = make_dataset(cfg, ns, "train"); val_full = make_dataset(cfg, ns, "val")
    train_set = Subset(train_full, range(min(ns.limit_train_episodes, len(train_full)))) if ns.limit_train_episodes else train_full
    val_limit = ns.limit_val_episodes or ns.validation_samples or int(cfg["training"]["validation_samples"])
    val_sampler = EpisodeBalancedSampler(val_full, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"], tuple(cfg["data"]["class_ids"]), int(cfg["project"]["seed"]) + 10_000_019, epoch_size=min(val_limit, len(val_full)))
    val_indices = list(iter(val_sampler))
    val_set = Subset(val_full, val_indices[rank::world])
    budget = ns.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    if isinstance(train_set, Subset):
        sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=int(cfg["project"]["seed"])) if world > 1 else None
    else:
        sampler = EpisodeBalancedSampler(train_set, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"], tuple(cfg["data"]["class_ids"]), int(cfg["project"]["seed"]), rank, world, budget)
    workers = ns.num_workers if ns.num_workers is not None else int(cfg["training"]["num_workers"])
    batch_size = ns.batch_size_per_gpu or int(cfg["training"]["batch_size_per_gpu"])
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, collate_fn=collate_prompt_episodes)
    if workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=sampler is None, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = PromptRegionModel(int(cfg["model"]["region_dim"]), int(cfg["model"]["heads"]), int(cfg["model"]["set_layers"]), float(cfg["model"]["dropout"])).to(device)
    transfer = load_fine_norm_from_phase4(model, str(ns.phase4_checkpoint))
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = ns.max_epochs or int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_fp16 = cfg["training"]["amp_dtype"] == "float16"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    start_epoch, history = 0, []
    if ns.resume:
        payload = torch.load(ns.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1; history = list(payload.get("history", []))
    end_epoch = epochs if ns.stop_after_epoch is None else min(epochs, ns.stop_after_epoch + 1)
    if end_epoch <= start_epoch:
        raise ValueError(f"stop-after-epoch={ns.stop_after_epoch} precedes start_epoch={start_epoch}")

    stamp = ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["output"]["root"]) / f"phase5_prompt_encoder_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True)
        save_json(output / "run_metadata.json", {"timestamp": stamp, "test_used": False, "world_size": world, "start_epoch": start_epoch, "epochs": epochs, "end_epoch_exclusive": end_epoch, "stop_after_epoch": ns.stop_after_epoch, "batch_size_per_gpu": batch_size, "num_workers": workers, "samples_per_epoch": budget, "validation_samples": len(val_indices), "reproducibility": {"command": [sys.executable, *sys.argv], "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "visible_gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())], "source_sha256": source_hashes()}, "sampler": {"type": "class_size_balanced_group_conditional", "empty_train_buckets": getattr(sampler, "empty_buckets", []), "empty_train_group_buckets": getattr(sampler, "empty_group_buckets", []), "empty_val_buckets": val_sampler.empty_buckets, "empty_val_group_buckets": val_sampler.empty_group_buckets}, "inputs": {"cache_index": str(ns.cache_index), "label_index": str(ns.label_index), "patch_index": str(ns.patch_index), "eligibility_index": str(ns.eligibility_index), "phase4_checkpoint": str(ns.phase4_checkpoint), "resume": str(ns.resume) if ns.resume else None}, "phase4_transfer": transfer, "config": cfg})
    if world > 1: dist.barrier()

    ignore = int(cfg["data"]["ignore_index"])
    for epoch in range(start_epoch, end_epoch):
        if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
        train_full.set_epoch(epoch); model.train(); sums = {"loss": 0.0, "balanced_bce": 0.0, "dice_loss": 0.0, "ranking_loss": 0.0}; steps = 0; t0 = time.monotonic()
        for batch in train_loader:
            batch = move_batch(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                output_values = model(batch)
                loss, parts = prompt_region_loss(output_values, batch["binary_target"], batch["prompted_regions"], ignore, float(cfg["loss"]["dice_weight"]), float(cfg["loss"]["ranking_weight"]), float(cfg["loss"]["ranking_margin"]))
            finite = torch.isfinite(loss).item() and finite_outputs(output_values) and all(torch.isfinite(v).item() for v in parts.values())
            if not all_ranks_true(finite, device, world): raise FloatingPointError(f"non-finite activation/loss epoch={epoch} step={steps} rank={rank} parts={parts}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
            if not all_ranks_true(torch.isfinite(grad).item(), device, world): raise FloatingPointError(f"non-finite gradient epoch={epoch} step={steps} rank={rank}")
            scaler.step(optimizer); scaler.update(); steps += 1; sums["loss"] += float(loss.detach())
            for name, value in parts.items(): sums[name] += float(value)
            if rank == 0 and ns.log_every_steps and steps % ns.log_every_steps == 0: print({"event": "train_progress", "epoch": epoch, "step": steps, "loss": sums["loss"] / steps}, flush=True)
        scheduler.step()

        model.eval(); count_names = ("tp", "fp", "fn", "tn", "unprompted_tp", "unprompted_positive")
        val_sums = {"loss": 0.0, "steps": 0.0, **{name: 0.0 for name in count_names}}
        stratum_names = [f"class_{class_id:02d}" for class_id in cfg["data"]["class_ids"]] + [f"size_{name}" for name in ("point", "small", "large")]
        val_strata = {name: {count: 0.0 for count in count_names} for name in stratum_names}
        validation_finite = True
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=amp_dtype):
                    prediction = model(batch); val_loss, _ = prompt_region_loss(prediction, batch["binary_target"], batch["prompted_regions"], ignore, float(cfg["loss"]["dice_weight"]), float(cfg["loss"]["ranking_weight"]), float(cfg["loss"]["ranking_margin"]))
                validation_finite = validation_finite and torch.isfinite(val_loss).item() and finite_outputs(prediction)
                val_sums["loss"] += float(val_loss); val_sums["steps"] += 1
                for name, value in metric_counts(prediction["logits"], batch["binary_target"], batch["prompted_regions"], ignore).items(): val_sums[name] += float(value)
                masks = {
                    **{f"class_{class_id:02d}": batch["target_class"] == class_id for class_id in cfg["data"]["class_ids"]},
                    **{f"size_{name}": batch["prompt_size_id"] == size_id for size_id, name in enumerate(("point", "small", "large"))},
                }
                for stratum, selected in masks.items():
                    if selected.any():
                        counts = metric_counts(prediction["logits"][selected], batch["binary_target"][selected], batch["prompted_regions"][selected], ignore)
                        for name, value in counts.items(): val_strata[stratum][name] += float(value)
        if not all_ranks_true(validation_finite, device, world):
            raise FloatingPointError(f"non-finite validation epoch={epoch} rank={rank}")
        train_names = [f"train_{name}" for name in sums] + ["train_steps"]
        val_names = [f"val_{name}" for name in val_sums]
        stratum_value_names = [f"stratum_{stratum}_{name}" for stratum in stratum_names for name in count_names]
        names = train_names + val_names + stratum_value_names
        values = torch.tensor([sums[n] for n in sums] + [steps] + [val_sums[n] for n in val_sums] + [val_strata[stratum][name] for stratum in stratum_names for name in count_names], dtype=torch.float64, device=device)
        if world > 1: dist.all_reduce(values)
        if rank == 0:
            reduced = dict(zip(names, values.tolist())); metric_input = {n: reduced[f"val_{n}"] for n in ("tp", "fp", "fn", "tn", "unprompted_tp", "unprompted_positive")}
            stratum_metrics = {stratum: metrics_from_counts({name: reduced[f"stratum_{stratum}_{name}"] for name in count_names}) for stratum in stratum_names}
            row = {"epoch": epoch, "train_loss": reduced["train_loss"] / max(reduced["train_steps"], 1), "train_balanced_bce": reduced["train_balanced_bce"] / max(reduced["train_steps"], 1), "train_dice_loss": reduced["train_dice_loss"] / max(reduced["train_steps"], 1), "train_ranking_loss": reduced["train_ranking_loss"] / max(reduced["train_steps"], 1), "val_loss": reduced["val_loss"] / max(reduced["val_steps"], 1), **metrics_from_counts(metric_input), "val_by_class": {name: stratum_metrics[name] for name in stratum_names if name.startswith("class_")}, "val_by_size": {name: stratum_metrics[name] for name in stratum_names if name.startswith("size_")}, "lr": optimizer.param_groups[0]["lr"], "epoch_elapsed_sec": round(time.monotonic() - t0, 2)}
            history.append(row); ckpt_path = output / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model": (model.module if world > 1 else model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": cfg, "history": history}, ckpt_path)
            save_json(output / "last_checkpoint.json", {"epoch": epoch, "path": str(ckpt_path)})
            best = max(history, key=lambda item: item["region_dice"]); save_json(output / "best_checkpoint.json", {"epoch": best["epoch"], "path": str(output / f"checkpoint_epoch_{best['epoch']:03d}.pth"), "region_dice": best["region_dice"]})
            save_json(output / "metrics.json", history); print(row, flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
