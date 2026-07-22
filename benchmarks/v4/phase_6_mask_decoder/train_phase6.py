#!/usr/bin/env python3
"""Train the Phase-6 residual region-graph mask decoder."""
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

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import (
    EpisodeBalancedSampler,
    PromptEpisodeDataset,
    collate_prompt_episodes,
)
from benchmarks.v4.phase_5_prompt_encoder.src.losses import metric_counts, metrics_from_counts
from benchmarks.v4.phase_6_mask_decoder.src.losses import decoder_loss
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder, load_phase5_prompt_model


TENSOR_KEYS = (
    "prompt_size_id", "target_class", "fine_tokens", "fine_active", "region_xy", "region_area",
    "positive_tokens", "negative_tokens", "positive_xy", "negative_xy", "positive_mask",
    "negative_mask", "prompted_regions", "binary_target",
)
COUNT_NAMES = ("tp", "fp", "fn", "tn", "unprompted_tp", "unprompted_positive")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--limit-train-episodes", type=int)
    parser.add_argument("--limit-val-episodes", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every-steps", type=int, default=50)
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path("benchmarks/v4/phase_6_mask_decoder/src/model.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/losses.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/configs/phase6_mask_decoder.yaml"),
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
    return all(torch.isfinite(value).all().item() for value in output.values() if value.is_floating_point())


def make_dataset(cfg: dict, args, split: str) -> PromptEpisodeDataset:
    return PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, split,
        seed=int(cfg["project"]["seed"]),
        size_probabilities=cfg["data"]["size_probabilities"],
        target_class_ids=tuple(cfg["data"]["class_ids"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
        centroid_knn=int(cfg["data"]["centroid_knn"]),
        eligibility_index=args.eligibility_index,
    )


def accumulate_counts(destination: dict[str, float], source: dict[str, torch.Tensor], prefix: str) -> None:
    for name, value in source.items():
        destination[f"{prefix}_{name}"] += float(value)


def main() -> None:
    args = parse_args(); cfg = load_config(args.config)
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-6 training requires CUDA")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    seed_all(int(cfg["project"]["seed"]) + rank)

    train_full = make_dataset(cfg, args, "train"); val_full = make_dataset(cfg, args, "val")
    train_set = Subset(train_full, range(min(args.limit_train_episodes, len(train_full)))) if args.limit_train_episodes else train_full
    val_limit = args.limit_val_episodes or args.validation_samples or int(cfg["training"]["validation_samples"])
    val_sampler = EpisodeBalancedSampler(
        val_full, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"],
        tuple(cfg["data"]["class_ids"]), int(cfg["project"]["seed"]) + 10_000_019,
        epoch_size=min(val_limit, len(val_full)),
    )
    val_indices = list(iter(val_sampler)); val_set = Subset(val_full, val_indices[rank::world])
    budget = args.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    if isinstance(train_set, Subset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, world, rank, shuffle=True, seed=int(cfg["project"]["seed"])
        ) if world > 1 else None
    else:
        sampler = EpisodeBalancedSampler(
            train_set, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"],
            tuple(cfg["data"]["class_ids"]), int(cfg["project"]["seed"]), rank, world, budget,
        )
    workers = args.num_workers if args.num_workers is not None else int(cfg["training"]["num_workers"])
    batch_size = args.batch_size_per_gpu or int(cfg["training"]["batch_size_per_gpu"])
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, collate_fn=collate_prompt_episodes)
    if workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=sampler is None, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model_cfg = cfg["model"]
    prompt_model, transfer = load_phase5_prompt_model(
        str(args.phase5_checkpoint), int(model_cfg["region_dim"]), int(model_cfg["prompt_heads"]),
        int(model_cfg["prompt_set_layers"]), float(model_cfg["prompt_dropout"]),
    )
    model = ContextAwareMaskDecoder(
        prompt_model, int(model_cfg["region_dim"]), int(model_cfg["graph_heads"]),
        int(model_cfg["graph_layers"]), int(model_cfg["graph_neighbours"]),
        float(model_cfg["graph_dropout"]), bool(model_cfg["freeze_prompt_encoder"]),
        float(model_cfg["residual_limit"]),
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("decoder has no trainable parameters")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = args.max_epochs or int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_fp16 = cfg["training"]["amp_dtype"] == "float16"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    start_epoch, history = 0, []
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1; history = list(payload.get("history", []))
    end_epoch = epochs if args.stop_after_epoch is None else min(epochs, args.stop_after_epoch + 1)
    if end_epoch <= start_epoch:
        raise ValueError(f"stop-after-epoch={args.stop_after_epoch} precedes start_epoch={start_epoch}")

    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["output"]["root"]) / f"phase6_mask_decoder_{stamp}"
    if not all_ranks_true(not output.exists(), device, world):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True)
        save_json(output / "run_metadata.json", {
            "timestamp": stamp, "test_used": False, "world_size": world, "start_epoch": start_epoch,
            "epochs": epochs, "end_epoch_exclusive": end_epoch, "stop_after_epoch": args.stop_after_epoch,
            "batch_size_per_gpu": batch_size, "num_workers": workers, "samples_per_epoch": budget,
            "validation_samples": len(val_indices), "trainable_parameter_count": sum(p.numel() for p in trainable),
            "total_parameter_count": sum(p.numel() for p in (model.module if world > 1 else model).parameters()),
            "reproducibility": {"command": [sys.executable, *sys.argv], "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "source_sha256": source_hashes()},
            "inputs": {"cache_index": str(args.cache_index), "label_index": str(args.label_index), "patch_index": str(args.patch_index), "eligibility_index": str(args.eligibility_index), "phase5_checkpoint": str(args.phase5_checkpoint), "resume": str(args.resume) if args.resume else None},
            "phase5_transfer": transfer, "config": cfg,
        })
    if world > 1:
        dist.barrier()

    ignore = int(cfg["data"]["ignore_index"]); loss_cfg = cfg["loss"]
    strata = [f"class_{class_id:02d}" for class_id in cfg["data"]["class_ids"]] + [f"size_{name}" for name in ("point", "small", "large")]
    for epoch in range(start_epoch, end_epoch):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_full.set_epoch(epoch); model.train(); t0 = time.monotonic(); steps = 0
        train_sums = {name: 0.0 for name in ("loss", "balanced_bce", "dice_loss", "ranking_loss", "boundary_loss")}
        for batch in train_loader:
            batch = move_batch(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                prediction = model(batch)
                loss, parts = decoder_loss(
                    prediction, batch["binary_target"], batch["prompted_regions"], ignore,
                    float(loss_cfg["dice_weight"]), float(loss_cfg["ranking_weight"]),
                    float(loss_cfg["ranking_margin"]), float(loss_cfg["boundary_weight"]),
                )
            finite = torch.isfinite(loss).item() and finite_outputs(prediction) and all(torch.isfinite(v).item() for v in parts.values())
            if not all_ranks_true(finite, device, world):
                raise FloatingPointError(f"non-finite activation/loss epoch={epoch} step={steps} rank={rank} parts={parts}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(trainable, float(cfg["training"]["max_grad_norm"]))
            if not all_ranks_true(torch.isfinite(gradient).item(), device, world):
                raise FloatingPointError(f"non-finite gradient epoch={epoch} step={steps} rank={rank}")
            scaler.step(optimizer); scaler.update(); steps += 1; train_sums["loss"] += float(loss.detach())
            for name, value in parts.items():
                train_sums[name] += float(value)
            if rank == 0 and args.log_every_steps and steps % args.log_every_steps == 0:
                print({"event": "train_progress", "epoch": epoch, "step": steps, "loss": train_sums["loss"] / steps}, flush=True)
        scheduler.step()

        model.eval(); totals = {f"{kind}_{name}": 0.0 for kind in ("initial", "refined") for name in COUNT_NAMES}
        totals.update({"loss": 0.0, "steps": 0.0})
        stratum_counts = {s: {f"{kind}_{name}": 0.0 for kind in ("initial", "refined") for name in COUNT_NAMES} for s in strata}
        validation_finite = True
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=amp_dtype):
                    prediction = model(batch)
                    val_loss, _ = decoder_loss(
                        prediction, batch["binary_target"], batch["prompted_regions"], ignore,
                        float(loss_cfg["dice_weight"]), float(loss_cfg["ranking_weight"]),
                        float(loss_cfg["ranking_margin"]), float(loss_cfg["boundary_weight"]),
                    )
                validation_finite &= torch.isfinite(val_loss).item() and finite_outputs(prediction)
                totals["loss"] += float(val_loss); totals["steps"] += 1
                accumulate_counts(totals, metric_counts(prediction["initial_logits"], batch["binary_target"], batch["prompted_regions"], ignore), "initial")
                accumulate_counts(totals, metric_counts(prediction["logits"], batch["binary_target"], batch["prompted_regions"], ignore), "refined")
                masks = {
                    **{f"class_{class_id:02d}": batch["target_class"] == class_id for class_id in cfg["data"]["class_ids"]},
                    **{f"size_{name}": batch["prompt_size_id"] == size_id for size_id, name in enumerate(("point", "small", "large"))},
                }
                for stratum, selected in masks.items():
                    if selected.any():
                        for kind, logits in (("initial", prediction["initial_logits"]), ("refined", prediction["logits"])):
                            accumulate_counts(stratum_counts[stratum], metric_counts(logits[selected], batch["binary_target"][selected], batch["prompted_regions"][selected], ignore), kind)
        if not all_ranks_true(validation_finite, device, world):
            raise FloatingPointError(f"non-finite validation epoch={epoch} rank={rank}")
        names = (
            [f"train_{name}" for name in train_sums] + ["train_steps"] + list(totals)
            + [f"stratum_{s}_{name}" for s in strata for name in stratum_counts[s]]
        )
        raw = (
            list(train_sums.values()) + [steps] + [totals[name] for name in totals]
            + [stratum_counts[s][name] for s in strata for name in stratum_counts[s]]
        )
        values = torch.tensor(raw, dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(values)
        if rank == 0:
            reduced = dict(zip(names, values.tolist()))
            initial = metrics_from_counts({name: reduced[f"initial_{name}"] for name in COUNT_NAMES})
            refined = metrics_from_counts({name: reduced[f"refined_{name}"] for name in COUNT_NAMES})
            by_stratum = {}
            for stratum in strata:
                by_stratum[stratum] = {
                    kind: metrics_from_counts({name: reduced[f"stratum_{stratum}_{kind}_{name}"] for name in COUNT_NAMES})
                    for kind in ("initial", "refined")
                }
            row = {
                "epoch": epoch,
                **{f"train_{name}": reduced[f"train_{name}"] / max(reduced["train_steps"], 1) for name in train_sums},
                "val_loss": reduced["loss"] / max(reduced["steps"], 1),
                "initial": initial, "refined": refined,
                "region_dice_gain": refined["region_dice"] - initial["region_dice"],
                "val_by_class": {k: v for k, v in by_stratum.items() if k.startswith("class_")},
                "val_by_size": {k: v for k, v in by_stratum.items() if k.startswith("size_")},
                "lr": optimizer.param_groups[0]["lr"], "epoch_elapsed_sec": round(time.monotonic() - t0, 2),
            }
            history.append(row); checkpoint = output / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model": (model.module if world > 1 else model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": cfg, "history": history}, checkpoint)
            save_json(output / "last_checkpoint.json", {"epoch": epoch, "path": str(checkpoint)})
            best = max(history, key=lambda item: item["refined"]["region_dice"])
            save_json(output / "best_checkpoint.json", {"epoch": best["epoch"], "path": str(output / f"checkpoint_epoch_{best['epoch']:03d}.pth"), "region_dice": best["refined"]["region_dice"], "region_dice_gain": best["region_dice_gain"]})
            save_json(output / "metrics.json", history); print(row, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
