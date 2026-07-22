#!/usr/bin/env python3
"""Joint pixel-supervised fine-only Phase-2→3→5→6 training."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import EpisodeBalancedSampler, PromptEpisodeDataset
from benchmarks.v4.phase_5_prompt_encoder.src.losses import metric_counts, metrics_from_counts
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts, boundary_f1, dice_from_counts
from benchmarks.v4.phase_6_mask_decoder.src.conflict_policy import (
    STRESS_COLUMNS,
    conflict_free_training_batch,
    conflict_stress_rows,
)
from benchmarks.v4.phase_6_mask_decoder.src.checkpoint_policy import (
    best_checkpoint_pointer,
    build_pareto_report,
)
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import JointPixelEpisodeDataset, collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import joint_pixel_loss
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import (
    FrozenPromptTeacher,
    FrozenRegionGeometryTeacher,
    JointPromptMaskModel,
    load_joint_components,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--train-cell-routing", type=Path, required=True)
    parser.add_argument("--val-cell-routing", type=Path, required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-joint-checkpoint", type=Path)
    parser.add_argument("--geometry-teacher-checkpoint", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--overfit-episode-index", type=int)
    parser.add_argument("--gradient-audit-first-step", action="store_true")
    parser.add_argument("--prompt-conflict-margin-weight", type=float)
    parser.add_argument("--prompt-geometry-anchor-weight", type=float)
    parser.add_argument("--log-every-steps", type=int, default=25)
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = [
        Path(__file__), Path("benchmarks/v4/phase_6_mask_decoder/src/joint_dataset.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/joint_model.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/joint_losses.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/model.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/conflict_policy.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/checkpoint_policy.py"),
        Path("benchmarks/v4/phase_4_cross_scale/src/model.py"),
        Path("benchmarks/v4/phase_2_region_encoder/src/model.py"),
        Path(config_path),
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
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def make_episodes(cfg: dict, args, split: str) -> PromptEpisodeDataset:
    return PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, split,
        seed=int(cfg["project"]["seed"]), size_probabilities=cfg["data"]["size_probabilities"],
        target_class_ids=tuple(cfg["data"]["class_ids"]), ignore_index=int(cfg["data"]["ignore_index"]),
        centroid_knn=int(cfg["data"]["centroid_knn"]), eligibility_index=args.eligibility_index,
    )


def gradient_norm(parameters: list[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    values = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.zeros((), device=device)


def component_gradient_audit(
    output: dict,
    batch: dict,
    ignore: int,
    configured_weights: dict[str, float],
    region_ranking_margin: float,
    groups: dict[str, list[torch.nn.Parameter]],
) -> dict:
    """Measure each top-level objective on one shared forward graph."""
    component_names = (
        "pixel_bce", "pixel_dice", "pixel_boundary", "region_aux",
        "assignment_balance", "assignment_entropy", "assignment_compactness",
        "prompt_separation", "prompt_conflict_margin", "prompt_sign",
        "prompt_geometry_anchor", "teacher_logit", "teacher_task",
    )
    zero_weights = {name: 0.0 for name in configured_weights}
    flat_parameters = [parameter for parameters in groups.values() for parameter in parameters]
    rows = {}
    for component in (*component_names, "configured_total"):
        if component == "prompt_geometry_anchor" and "geometry_teacher_region_xy" not in output:
            rows[component] = {"available": False, "reason": "geometry teacher not configured"}
            continue
        weights = dict(configured_weights) if component == "configured_total" else dict(zero_weights)
        if component != "configured_total":
            weights[component] = 1.0
        loss, parts = joint_pixel_loss(output, batch, ignore, weights, float(region_ranking_margin))
        gradients = torch.autograd.grad(loss, flat_parameters, retain_graph=True, allow_unused=True)
        offset = 0; norms = {}
        for name, parameters in groups.items():
            local = gradients[offset:offset + len(parameters)]; offset += len(parameters)
            squares = [gradient.detach().float().square().sum() for gradient in local if gradient is not None]
            norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
        rows[component] = {
            "loss": float(loss.detach()), "gradient_norms": norms,
            "prompt_conflict_slots": int(parts["prompt_conflict_slots"]),
            "prompt_conflict_episodes": int(parts["prompt_conflict_episodes"]),
            "prompt_conflict_margin_pairs": int(parts["prompt_conflict_margin_pairs"]),
        }
    return {"components": rows, "configured_weights": configured_weights}


FIELDS = (
    "pixel_tp", "pixel_fp", "pixel_fn", "pixel_macro_sum", "pixel_macro_count",
    "region_tp", "region_fp", "region_fn", "region_macro_sum", "region_macro_count",
    "unprompted_tp", "unprompted_fp", "unprompted_fn", "unprompted_macro_sum", "unprompted_macro_count",
    "boundary_sum", "boundary_count",
    "prompt_conflict_slots", "prompt_conflict_episodes", "prompt_episodes",
)


def empty_validation(scopes: tuple[str, ...]) -> dict[str, float]:
    return {f"{scope}_{field}": 0.0 for scope in scopes for field in FIELDS}


def update_scope(values: dict[str, float], scope: str, selected: torch.Tensor, output: dict, batch: dict, ignore: int) -> None:
    if not selected.any():
        return
    pixel_truth = batch["pixel_gt"][selected] == batch["target_class"][selected][:, None, None]
    pixel_valid = batch["pixel_gt"][selected] != ignore
    pixel_prediction = output["pixel_probability"][selected] >= 0.5
    pixel = binary_counts(pixel_prediction, pixel_truth, pixel_valid)
    online_target = output["online_binary_target"][selected]
    region_truth = online_target == 1
    region_valid = online_target != ignore
    region_prediction = output["logits"][selected] >= 0
    region = binary_counts(region_prediction, region_truth, region_valid)
    query_valid = region_valid & ~output["online_all_prompted_regions"][selected]
    query = binary_counts(region_prediction, region_truth, query_valid)
    boundary = boundary_f1(pixel_prediction, pixel_truth, pixel_valid, tolerance=2)
    for prefix, counts in (("pixel", pixel), ("region", region), ("unprompted", query)):
        for name in ("tp", "fp", "fn"):
            values[f"{scope}_{prefix}_{name}"] += float(counts[name].sum())
        dice = torch.as_tensor(
            dice_from_counts(counts["tp"].cpu().numpy(), counts["fp"].cpu().numpy(), counts["fn"].cpu().numpy()),
            device=selected.device,
        )
        keep = torch.isfinite(dice)
        if prefix == "unprompted":
            keep &= counts["positive"] > 0
        values[f"{scope}_{prefix}_macro_sum"] += float(dice[keep].sum()) if keep.any() else 0.0
        values[f"{scope}_{prefix}_macro_count"] += float(keep.sum())
    finite_boundary = torch.isfinite(boundary["boundary_f1"])
    values[f"{scope}_boundary_sum"] += float(boundary["boundary_f1"][finite_boundary].sum()) if finite_boundary.any() else 0.0
    values[f"{scope}_boundary_count"] += float(finite_boundary.sum())
    conflicts = output["online_prompt_conflicts"][selected]
    values[f"{scope}_prompt_conflict_slots"] += float(conflicts.sum())
    values[f"{scope}_prompt_conflict_episodes"] += float(conflicts.any(1).sum())
    values[f"{scope}_prompt_episodes"] += float(selected.sum())


def scope_metrics(values: dict[str, float], scope: str) -> dict:
    result = {}
    for prefix in ("pixel", "region", "unprompted"):
        tp, fp, fn = (values[f"{scope}_{prefix}_{name}"] for name in ("tp", "fp", "fn"))
        result[f"{prefix}_micro_dice"] = float(dice_from_counts(tp, fp, fn))
        result[f"{prefix}_macro_dice"] = values[f"{scope}_{prefix}_macro_sum"] / max(values[f"{scope}_{prefix}_macro_count"], 1.0)
        result[f"{prefix}_evaluable_episodes"] = int(values[f"{scope}_{prefix}_macro_count"])
    result["boundary_f1"] = values[f"{scope}_boundary_sum"] / max(values[f"{scope}_boundary_count"], 1.0)
    result["boundary_evaluable_episodes"] = int(values[f"{scope}_boundary_count"])
    episodes = values[f"{scope}_prompt_episodes"]
    result["prompt_conflict_slots"] = int(values[f"{scope}_prompt_conflict_slots"])
    result["prompt_conflict_episodes"] = int(values[f"{scope}_prompt_conflict_episodes"])
    result["prompt_episodes"] = int(episodes)
    result["prompt_conflict_episode_rate"] = values[f"{scope}_prompt_conflict_episodes"] / max(episodes, 1.0)
    result["prompt_conflict_slots_per_episode"] = values[f"{scope}_prompt_conflict_slots"] / max(episodes, 1.0)
    return result


def initialization_audit(model: JointPromptMaskModel, output: dict, batch: dict, ignore: int, classes: int) -> dict:
    with torch.no_grad():
        cached = model.decoder.prompt_model(batch)["logits"]
        compared = batch["fine_active"]
        mismatch = ((output["online_binary_target"] != batch["binary_target"]) & compared).sum()
        positive_changed = (
            (output["online_positive_slot_indices"] != batch["positive_slot_indices"])
            & batch["positive_mask"]
        ).sum()
        negative_changed = (
            (output["online_negative_slot_indices"] != batch["negative_slot_indices"])
            & batch["negative_mask"]
        ).sum()
        audit = {
            "region_label_mismatch_slots": int(mismatch),
            "compared_slots": int(compared.sum()),
            "positive_prompt_slot_changes": int(positive_changed),
            "negative_prompt_slot_changes": int(negative_changed),
            "positive_negative_prompt_conflict_slots": int(output["online_prompt_conflicts"].sum()),
            "online_vs_cached_fused_token_max_abs": float((output["online_fused_tokens"] - batch["fine_tokens"]).abs().max()),
            "online_vs_cached_xy_max_abs": float((output["online_region_xy"] - batch["region_xy"]).abs().max()),
            "online_vs_cached_area_max_abs": float((output["online_region_area"] - batch["region_area"]).abs().max()),
            "online_vs_cached_phase5_logit_max_abs": float((output["initial_logits"] - cached).abs().max()),
        }
        if "geometry_teacher_region_xy" in output:
            audit.update({
                "geometry_teacher_prompt_conflict_slots": int(
                    output["geometry_teacher_prompt_conflicts"].sum()
                ),
                "online_vs_geometry_teacher_xy_max_abs": float(
                    (output["online_region_xy"] - output["geometry_teacher_region_xy"]).abs().max()
                ),
                "online_vs_geometry_teacher_area_max_abs": float(
                    (output["online_region_area"] - output["geometry_teacher_region_area"]).abs().max()
                ),
            })
        if "parent_context_gate" in output:
            audit.update({
                "parent_context_initial_token_max_abs": float(
                    (output["online_contextual_tokens"] - output["online_fused_tokens"]).abs().max()
                ),
                "parent_context_initial_gate": float(output["parent_context_gate"]),
            })
        return audit


def main() -> None:
    args = parse_args(); cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    if args.prompt_conflict_margin_weight is not None:
        if args.prompt_conflict_margin_weight < 0:
            raise ValueError("--prompt-conflict-margin-weight must be non-negative")
        cfg["loss"]["weights"]["prompt_conflict_margin"] = float(args.prompt_conflict_margin_weight)
    if args.prompt_geometry_anchor_weight is not None:
        if args.prompt_geometry_anchor_weight < 0:
            raise ValueError("--prompt-geometry-anchor-weight must be non-negative")
        cfg["loss"]["weights"]["prompt_geometry_anchor"] = float(args.prompt_geometry_anchor_weight)
    if args.resume is not None and args.initial_joint_checkpoint is not None:
        raise ValueError("--resume and --initial-joint-checkpoint are mutually exclusive")
    anchor_weight = float(cfg["loss"]["weights"].get("prompt_geometry_anchor", 0.0))
    if anchor_weight > 0 and args.geometry_teacher_checkpoint is None:
        raise ValueError(
            "positive prompt_geometry_anchor weight requires --geometry-teacher-checkpoint"
        )
    if anchor_weight == 0 and args.geometry_teacher_checkpoint is not None:
        raise ValueError(
            "--geometry-teacher-checkpoint requires a positive prompt_geometry_anchor weight"
        )
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("joint pixel training requires CUDA")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank); seed_all(int(cfg["project"]["seed"]) + rank)

    train_episodes = make_episodes(cfg, args, "train"); val_episodes = make_episodes(cfg, args, "val")
    parent_context = bool(cfg["model"].get("train_parent_context", False))
    train_set = JointPixelEpisodeDataset(train_episodes, args.patch_index, args.train_cell_routing, p2cfg["data"]["class_map"], int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]), include_parent_context=parent_context)
    val_full = JointPixelEpisodeDataset(val_episodes, args.patch_index, args.val_cell_routing, p2cfg["data"]["class_map"], int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]), include_parent_context=parent_context)
    budget = args.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    val_budget = args.validation_samples or int(cfg["training"]["validation_samples"])
    if args.overfit_episode_index is not None:
        if world != 1:
            raise ValueError("fixed-episode overfit mode is single-GPU only")
        if not 0 <= args.overfit_episode_index < len(train_set):
            raise IndexError(f"overfit episode index {args.overfit_episode_index} outside [0, {len(train_set)})")
        train_data = Subset(train_set, [args.overfit_episode_index])
        train_sampler = RandomSampler(train_data, replacement=True, num_samples=budget)
        val_indices = [args.overfit_episode_index]
        val_set = Subset(train_set, val_indices)
    else:
        train_data = train_set
        train_sampler = EpisodeBalancedSampler(
            train_episodes, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"], tuple(cfg["data"]["class_ids"]),
            int(cfg["project"]["seed"]), rank, world, budget,
        )
        val_sampler = EpisodeBalancedSampler(
            val_episodes, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"], tuple(cfg["data"]["class_ids"]),
            int(cfg["project"]["seed"]) + 10_000_019, epoch_size=min(val_budget, len(val_episodes)),
        )
        val_indices = list(iter(val_sampler)); val_set = Subset(val_full, val_indices[rank::world])
    batch_size = args.batch_size_per_gpu or int(cfg["training"]["batch_size_per_gpu"])
    workers = args.num_workers if args.num_workers is not None else int(cfg["training"]["num_workers"])
    collate = functools.partial(collate_joint_pixel_episodes, max_cells=int(cfg["data"]["max_cells_per_patch"]))
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, collate_fn=collate)
    if workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_data, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    phase2, cell, prompt, transfer = load_joint_components(
        p2cfg, args.phase2_checkpoint, args.cell_checkpoint, args.phase5_checkpoint, p5cfg, int(cfg["data"]["cell_feature_dim"]),
    )
    mc = cfg["model"]
    model = JointPromptMaskModel(
        phase2, cell, prompt, int(mc["region_dim"]), int(mc["graph_heads"]), int(mc["graph_layers"]),
        int(mc["graph_neighbours"]), float(mc["graph_dropout"]), float(mc["residual_limit"]),
        bool(mc["train_phase2_embedding"]), bool(mc["train_phase2_assignment"]), bool(mc["train_cell"]), bool(mc["train_prompt"]),
        bool(mc.get("train_decoder", True)), bool(mc.get("train_parent_context", False)),
        bool(mc.get("train_backbone_layer4", False)),
        len(p2cfg["data"]["class_map"]), int(cfg["data"]["ignore_index"]),
    ).to(device)
    initial_joint = None
    if args.initial_joint_checkpoint is not None:
        payload = torch.load(args.initial_joint_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {
            f"parent_context.{name}" for name in model.parent_context.state_dict()
        } if model.parent_context is not None else set()
        if unexpected or set(missing) - allowed_missing:
            raise ValueError(
                f"incompatible initial joint checkpoint missing={missing} unexpected={unexpected}"
            )
        initial_joint = {
            "checkpoint": str(args.initial_joint_checkpoint), "epoch": int(payload["epoch"]),
            "missing": list(missing), "unexpected": list(unexpected),
        }
    prompt_parameters = [p for p in model.decoder.prompt_model.parameters() if p.requires_grad]
    prompt_parameter_ids = {id(parameter) for parameter in prompt_parameters}
    groups = {
        "decoder": [p for p in model.decoder.parameters() if p.requires_grad and id(p) not in prompt_parameter_ids],
        "phase2_embedding": [p for p in model.phase2.embedding.parameters() if p.requires_grad],
        "phase2_assignment": [p for p in model.phase2.assignment.parameters() if p.requires_grad],
        "cell": [p for p in model.cell.parameters() if p.requires_grad],
        "prompt": prompt_parameters,
        "parent_context": [p for p in model.parent_context.parameters() if p.requires_grad] if model.parent_context is not None else [],
        "backbone_layer4": [p for p in model.phase2.backbone.layer4.parameters() if p.requires_grad] if bool(mc.get("train_backbone_layer4", False)) else [],
    }
    groups = {name: parameters for name, parameters in groups.items() if parameters}
    required_groups = set()
    required_groups.update(
        name for enabled, name in (
            (bool(mc["train_phase2_embedding"]), "phase2_embedding"),
            (bool(mc["train_phase2_assignment"]), "phase2_assignment"),
            (bool(mc["train_cell"]), "cell"),
            (bool(mc["train_prompt"]), "prompt"),
            (bool(mc.get("train_decoder", True)), "decoder"),
            (bool(mc.get("train_parent_context", False)), "parent_context"),
            (bool(mc.get("train_backbone_layer4", False)), "backbone_layer4"),
        ) if enabled
    )
    missing_groups = required_groups - set(groups)
    if missing_groups:
        raise ValueError(f"required trainable groups missing: {sorted(missing_groups)}")
    if not required_groups:
        raise ValueError("joint training requires at least one enabled trainable group")
    lr = {"decoder": "decoder_lr", "phase2_embedding": "phase2_embedding_lr", "phase2_assignment": "phase2_assignment_lr", "cell": "cell_lr", "prompt": "prompt_lr", "parent_context": "parent_context_lr", "backbone_layer4": "backbone_layer4_lr"}
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "lr": float(cfg["training"][lr[name]]), "name": name} for name, parameters in groups.items()],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    prompt_teacher = (
        FrozenPromptTeacher(model.decoder.prompt_model).to(device)
        if bool(mc["train_prompt"]) else None
    )
    if world > 1:
        # Every mutable BatchNorm is held in eval mode. Disabling per-forward
        # buffer broadcasts also lets validation ranks safely have unequal
        # final shard lengths (for example 4000 episodes over 6 ranks).
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    raw_model = model.module if world > 1 else model
    epochs = args.max_epochs or int(cfg["training"]["epochs"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_fp16 = cfg["training"]["amp_dtype"] == "float16"; amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16); start_epoch = 0; history = []
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False); raw_model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1; history = list(payload.get("history", []))
        for previous in history:
            if int(previous["epoch"]) == int(payload["epoch"]):
                previous.setdefault("checkpoint_path", str(args.resume))
    geometry_teacher = None
    geometry_teacher_transfer = None
    if args.geometry_teacher_checkpoint is not None:
        geometry_teacher = FrozenRegionGeometryTeacher(
            raw_model.phase2, args.geometry_teacher_checkpoint
        ).to(device)
        geometry_teacher_transfer = geometry_teacher.transfer
    end_epoch = epochs if args.stop_after_epoch is None else min(epochs, args.stop_after_epoch + 1)
    if end_epoch <= start_epoch:
        raise ValueError("requested end epoch does not follow resume epoch")

    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"); output_dir = Path(cfg["output"]["root"]) / f"phase6_joint_pixel_{stamp}"
    if not all_ranks_true(not output_dir.exists(), device, world):
        raise FileExistsError(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True)
        save_json(output_dir / "run_metadata.json", {
            "timestamp": stamp, "test_used": False, "world_size": world, "start_epoch": start_epoch, "epochs": epochs,
            "end_epoch_exclusive": end_epoch, "batch_size_per_gpu": batch_size, "num_workers": workers,
            "samples_per_epoch": budget, "validation_samples": len(val_indices), "trainable_groups": {name: sum(p.numel() for p in parameters) for name, parameters in groups.items()},
            "required_gradient_groups": sorted(required_groups),
            "overfit_episode_index": args.overfit_episode_index,
            "inputs": {name: str(getattr(args, name)) if getattr(args, name) is not None else None for name in ("phase2_config", "phase5_config", "phase2_checkpoint", "cell_checkpoint", "phase5_checkpoint", "cache_index", "label_index", "patch_index", "eligibility_index", "train_cell_routing", "val_cell_routing", "resume", "initial_joint_checkpoint", "geometry_teacher_checkpoint")},
            "transfers": transfer, "initial_joint": initial_joint,
            "geometry_teacher": geometry_teacher_transfer, "config": cfg,
            "prompt_teacher": {
                "enabled": prompt_teacher is not None,
                "source": (
                    str(args.initial_joint_checkpoint or args.phase5_checkpoint)
                    if prompt_teacher is not None else None
                ),
                "snapshot_stage": (
                    "after_initial_model_load_before_optimizer"
                    if prompt_teacher is not None else None
                ),
            },
            "reproducibility": {"command": [sys.executable, *sys.argv], "source_sha256": source_hashes(args.config), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        })
    if world > 1:
        dist.barrier()

    if args.gradient_audit_first_step and world != 1:
        raise ValueError("--gradient-audit-first-step is single-GPU only")

    ignore = int(cfg["data"]["ignore_index"]); classes = len(p2cfg["data"]["class_map"]); audit_done = args.resume is not None
    exclude_train_conflicts = bool(cfg["training"].get("exclude_prompt_conflict_episodes", False))
    scopes = ("overall",) + tuple(f"class_{value:02d}" for value in cfg["data"]["class_ids"]) + tuple(f"size_{name}" for name in ("point", "small", "large"))
    gradient_audit_done = False
    for epoch in range(start_epoch, end_epoch):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_set.set_epoch(0 if args.overfit_episode_index is not None else epoch)
        raw_model.train(); started = time.monotonic()
        train_sums = {"loss": 0.0, **{name: 0.0 for name in ("pixel_bce", "pixel_dice_loss", "pixel_boundary_loss", "region_aux_loss", "region_valid_episodes", "region_positive_evaluable_episodes", "region_negative_evaluable_episodes", "region_ranking_evaluable_episodes", "assignment_balance", "assignment_entropy", "assignment_compactness", "prompt_separation_loss", "prompt_conflict_margin_loss", "prompt_conflict_margin_pairs", "prompt_geometry_anchor_loss", "prompt_geometry_anchor_episodes", "prompt_geometry_anchor_prompts", "prompt_sign_loss", "teacher_logit_loss", "teacher_task_loss", "teacher_stable_regions", "prompt_conflict_slots", "prompt_conflict_episodes", "prompt_episodes", "input_prompt_conflict_slots", "input_prompt_conflict_episodes", "input_prompt_episodes", "excluded_prompt_conflict_episodes", "fully_filtered_batches")}}
        train_steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(batch, geometry_teacher=geometry_teacher, prompt_teacher=prompt_teacher)
                raw_conflicts = output["online_prompt_conflicts"]
                input_conflict_episodes = raw_conflicts.any(1)
                train_sums["input_prompt_conflict_slots"] += float(raw_conflicts.sum())
                train_sums["input_prompt_conflict_episodes"] += float(input_conflict_episodes.sum())
                train_sums["input_prompt_episodes"] += float(raw_conflicts.shape[0])
                loss_output, loss_batch = output, batch
                if exclude_train_conflicts:
                    keep, loss_output, loss_batch = conflict_free_training_batch(output, batch)
                    train_sums["excluded_prompt_conflict_episodes"] += float((~keep).sum())
                    if not keep.any():
                        train_sums["fully_filtered_batches"] += 1.0
                        if rank == 0:
                            print({"event": "fully_filtered_train_batch", "epoch": epoch, "input_episodes": int(keep.numel())}, flush=True)
                        continue
                loss, parts = joint_pixel_loss(loss_output, loss_batch, ignore, cfg["loss"]["weights"], float(cfg["loss"]["region_ranking_margin"]))
            if args.gradient_audit_first_step and not gradient_audit_done:
                audit_values = component_gradient_audit(
                    loss_output, loss_batch, ignore, cfg["loss"]["weights"],
                    float(cfg["loss"]["region_ranking_margin"]), groups,
                )
                audit_values.update({"epoch": epoch, "step": train_steps, "rank": rank})
                save_json(output_dir / "gradient_audit.json", audit_values)
                print({"event": "gradient_audit", **audit_values}, flush=True)
                gradient_audit_done = True
            if not audit_done:
                audit = initialization_audit(raw_model, output, batch, ignore, classes)
                if world > 1:
                    audit_tensor = torch.tensor(list(audit.values()), dtype=torch.float64, device=device)
                    dist.all_reduce(audit_tensor[:2], op=dist.ReduceOp.SUM)
                    dist.all_reduce(audit_tensor[2:], op=dist.ReduceOp.MAX)
                    audit = dict(zip(audit, audit_tensor.tolist()))
                if rank == 0:
                    save_json(output_dir / "initialization_audit.json", audit); print({"event": "initialization_audit", **audit}, flush=True)
                audit_done = True
            finite = torch.isfinite(loss).item() and all(torch.isfinite(value).all().item() for key, value in output.items() if torch.is_tensor(value) and value.is_floating_point() and key in ("logits", "pixel_probability", "assignment_low")) and all(torch.isfinite(value).item() for value in parts.values())
            if not all_ranks_true(finite, device, world):
                raise FloatingPointError(f"non-finite joint output/loss epoch={epoch} step={train_steps} rank={rank} parts={parts}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norms = {name: gradient_norm(parameters, device) for name, parameters in groups.items()}
            grad_ok = (
                all(torch.isfinite(value).item() for value in norms.values())
                and all(norms[name].item() > 0 for name in required_groups)
            )
            if not all_ranks_true(grad_ok, device, world):
                raise FloatingPointError(f"invalid group gradient epoch={epoch} step={train_steps} rank={rank} norms={norms}")
            torch.nn.utils.clip_grad_norm_([p for parameters in groups.values() for p in parameters], float(cfg["training"]["max_grad_norm"]))
            scaler.step(optimizer); scaler.update(); train_steps += 1; train_sums["loss"] += float(loss.detach())
            for name in train_sums:
                if name != "loss" and name in parts:
                    train_sums[name] += float(parts[name])
            if rank == 0 and args.log_every_steps and train_steps % args.log_every_steps == 0:
                print({"event": "train_progress", "epoch": epoch, "step": train_steps, "loss": train_sums["loss"] / train_steps, "gradient_norms": {name: float(value) for name, value in norms.items()}}, flush=True)
        if train_steps == 0:
            raise RuntimeError(f"every training batch was fully filtered at epoch {epoch}")
        scheduler.step(); raw_model.eval(); validation = empty_validation(scopes); val_loss_sum = 0.0; val_steps = 0; stress_rows = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=amp_dtype):
                    output = model(batch, geometry_teacher=geometry_teacher, prompt_teacher=prompt_teacher); val_loss, _ = joint_pixel_loss(output, batch, ignore, cfg["loss"]["weights"], float(cfg["loss"]["region_ranking_margin"]))
                if not torch.isfinite(val_loss):
                    raise FloatingPointError(f"non-finite validation rank={rank} epoch={epoch}")
                val_loss_sum += float(val_loss); val_steps += 1
                stress_rows.extend(conflict_stress_rows(output, batch, epoch, rank))
                masks = {"overall": torch.ones(len(batch["patch_id"]), dtype=torch.bool, device=device)}
                masks.update({f"class_{value:02d}": batch["target_class"] == value for value in cfg["data"]["class_ids"]})
                masks.update({f"size_{name}": batch["prompt_size_id"] == index for index, name in enumerate(("point", "small", "large"))})
                for scope, selected in masks.items():
                    update_scope(validation, scope, selected, output, batch, ignore)
        stress_shard = output_dir / f"stress_set_epoch_{epoch:03d}_rank_{rank:02d}.parquet"
        pd.DataFrame(stress_rows, columns=STRESS_COLUMNS).to_parquet(stress_shard, index=False)
        names = [f"train_{name}" for name in train_sums] + ["train_steps", "val_loss", "val_steps"] + list(validation)
        values = torch.tensor(list(train_sums.values()) + [train_steps, val_loss_sum, val_steps] + list(validation.values()), dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(values)
        if rank == 0:
            reduced = dict(zip(names, values.tolist())); val_values = {name: reduced[name] for name in validation}
            overall = scope_metrics(val_values, "overall")
            row = {
                "epoch": epoch, **{f"train_{name}": reduced[f"train_{name}"] / max(reduced["train_steps"], 1) for name in train_sums},
                "val_loss": reduced["val_loss"] / max(reduced["val_steps"], 1), **overall,
                "val_by_class": {scope: scope_metrics(val_values, scope) for scope in scopes if scope.startswith("class_")},
                "val_by_size": {scope: scope_metrics(val_values, scope) for scope in scopes if scope.startswith("size_")},
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "epoch_elapsed_sec": round(time.monotonic() - started, 2),
            }
            if raw_model.parent_context is not None:
                row["parent_context_gate"] = float(torch.tanh(raw_model.parent_context.gate.detach()))
            train_prompt_episodes = reduced["train_prompt_episodes"]
            row.update({
                "train_region_valid_episodes": int(reduced["train_region_valid_episodes"]),
                "train_region_positive_evaluable_episodes": int(reduced["train_region_positive_evaluable_episodes"]),
                "train_region_negative_evaluable_episodes": int(reduced["train_region_negative_evaluable_episodes"]),
                "train_region_ranking_evaluable_episodes": int(reduced["train_region_ranking_evaluable_episodes"]),
                "train_prompt_conflict_margin_pairs": int(reduced["train_prompt_conflict_margin_pairs"]),
                "train_prompt_geometry_anchor_episodes": int(reduced["train_prompt_geometry_anchor_episodes"]),
                "train_prompt_geometry_anchor_prompts": int(reduced["train_prompt_geometry_anchor_prompts"]),
                "train_prompt_conflict_slots": int(reduced["train_prompt_conflict_slots"]),
                "train_prompt_conflict_episodes": int(reduced["train_prompt_conflict_episodes"]),
                "train_prompt_episodes": int(train_prompt_episodes),
                "train_prompt_conflict_episode_rate": reduced["train_prompt_conflict_episodes"] / max(train_prompt_episodes, 1.0),
                "train_prompt_conflict_slots_per_episode": reduced["train_prompt_conflict_slots"] / max(train_prompt_episodes, 1.0),
                "train_input_prompt_conflict_slots": int(reduced["train_input_prompt_conflict_slots"]),
                "train_input_prompt_conflict_episodes": int(reduced["train_input_prompt_conflict_episodes"]),
                "train_input_prompt_episodes": int(reduced["train_input_prompt_episodes"]),
                "train_excluded_prompt_conflict_episodes": int(reduced["train_excluded_prompt_conflict_episodes"]),
                "train_fully_filtered_batches": int(reduced["train_fully_filtered_batches"]),
            })
            stress = pd.concat(
                [pd.read_parquet(output_dir / f"stress_set_epoch_{epoch:03d}_rank_{item:02d}.parquet") for item in range(world)],
                ignore_index=True,
            )
            expected_stress = int(overall["prompt_conflict_episodes"])
            if len(stress) != expected_stress:
                raise RuntimeError(f"stress set has {len(stress)} rows, expected {expected_stress}")
            stress_path = output_dir / f"stress_set_epoch_{epoch:03d}.parquet"
            stress.to_parquet(stress_path, index=False)
            row["stress_set_path"] = str(stress_path)
            row["stress_set_episodes"] = len(stress)
            checkpoint = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            row["checkpoint_path"] = str(checkpoint); history.append(row)
            torch.save({"epoch": epoch, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": cfg, "history": history}, checkpoint)
            save_json(output_dir / "last_checkpoint.json", {"epoch": epoch, "path": str(checkpoint)})
            pareto = build_pareto_report(
                history,
                cfg["checkpoint_selection"]["hard_gates"],
                cfg["checkpoint_selection"]["soft_targets"],
                cfg["checkpoint_selection"]["dice_reference"],
                cfg["checkpoint_selection"].get("noninferiority"),
            )
            save_json(output_dir / "pareto_checkpoints.json", pareto)
            save_json(output_dir / "best_checkpoint.json", best_checkpoint_pointer(pareto))
            save_json(output_dir / "metrics.json", history); print(row, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
