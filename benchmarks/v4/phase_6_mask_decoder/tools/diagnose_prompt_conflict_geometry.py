#!/usr/bin/env python3
"""Audit hard prompt conflicts and their soft centroid-distance surrogates."""
from __future__ import annotations

import argparse
import functools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import EpisodeBalancedSampler, PromptEpisodeDataset
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import JointPixelEpisodeDataset, collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.tools.evaluate_visualize_joint_pixel import construct_model, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--val-cell-routing", type=Path, required=True)
    parser.add_argument("--train-cell-routing", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--validation-episodes", type=int, default=360)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase6/evaluation"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def distribution_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)), "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def soft_overlap(
    positive_distance: torch.Tensor,
    negative_distance: torch.Tensor,
    positive_valid: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive = torch.softmax(-positive_distance / temperature, -1) * positive_valid.unsqueeze(-1)
    negative = torch.softmax(-negative_distance / temperature, -1) * negative_valid.unsqueeze(-1)
    positive = positive.sum(1) / positive_valid.sum(1, keepdim=True).clamp_min(1)
    negative = negative.sum(1) / negative_valid.sum(1, keepdim=True).clamp_min(1)
    return (positive * negative).sum(1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("prompt conflict diagnosis requires CUDA")
    device = torch.device("cuda", 0); torch.cuda.set_device(device)
    cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"prompt_conflict_geometry_{args.split}_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, args.split, int(cfg["project"]["seed"]),
        cfg["data"]["size_probabilities"], tuple(cfg["data"]["class_ids"]), int(cfg["data"]["ignore_index"]),
        int(cfg["data"]["centroid_knn"]), args.eligibility_index,
    )
    sampler_seed = int(cfg["project"]["seed"])
    if args.split == "val":
        sampler_seed += 10_000_019
    sampler = EpisodeBalancedSampler(
        episodes, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"],
        tuple(cfg["data"]["class_ids"]), sampler_seed,
        epoch_size=min(int(args.validation_episodes), len(episodes)),
    )
    selected = list(iter(sampler))
    cell_routing = args.val_cell_routing if args.split == "val" else args.train_cell_routing
    if cell_routing is None:
        raise ValueError("--train-cell-routing is required for --split train")
    full = JointPixelEpisodeDataset(
        episodes, args.patch_index, cell_routing, p2cfg["data"]["class_map"],
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]),
    )
    collate = functools.partial(collate_joint_pixel_episodes, max_cells=int(cfg["data"]["max_cells_per_patch"]))
    loader_args = {
        "batch_size": int(args.batch_size_per_gpu), "num_workers": int(args.num_workers),
        "pin_memory": True, "collate_fn": collate,
    }
    if args.num_workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(Subset(full, selected), shuffle=False, **loader_args)

    summaries = []; reference_slots = None
    for checkpoint in args.joint_checkpoint:
        model, load_report = construct_model(cfg, p2cfg, p5cfg, args, device, checkpoint)
        overlaps = {temperature: [] for temperature in (0.01, 0.05, 0.1)}
        conflict_overlaps = {temperature: [] for temperature in overlaps}
        nonconflict_overlaps = {temperature: [] for temperature in overlaps}
        conflict_margins = []; conflict_episodes = 0; conflict_slots = 0; slot_rows = []
        conflict_dataset_indices = []; selected_offset = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(batch)
                conflict = result["online_prompt_conflicts"]
                episode_conflict = conflict.any(1)
                conflict_episodes += int(episode_conflict.sum()); conflict_slots += int(conflict.sum())
                conflict_dataset_indices.extend(
                    int(selected[selected_offset + position])
                    for position in episode_conflict.nonzero(as_tuple=False).flatten().cpu().tolist()
                )
                selected_offset += len(episode_conflict)
                positive_distance = torch.cdist(batch["positive_xy"].float(), result["online_region_xy"].float())
                negative_distance = torch.cdist(batch["negative_xy"].float(), result["online_region_xy"].float())
                for temperature in overlaps:
                    values = soft_overlap(
                        positive_distance, negative_distance, batch["positive_mask"], batch["negative_mask"], temperature
                    )
                    overlaps[temperature].extend(values.cpu().tolist())
                    conflict_overlaps[temperature].extend(values[episode_conflict].cpu().tolist())
                    nonconflict_overlaps[temperature].extend(values[~episode_conflict].cpu().tolist())
                for distance, valid, slots in (
                    (positive_distance, batch["positive_mask"], result["online_positive_slot_indices"]),
                    (negative_distance, batch["negative_mask"], result["online_negative_slot_indices"]),
                ):
                    nearest_two = distance.topk(2, dim=-1, largest=False).values
                    margin = nearest_two[..., 1] - nearest_two[..., 0]
                    involved = valid & conflict.gather(1, slots.clamp_min(0))
                    conflict_margins.extend(margin[involved].cpu().tolist())
                for positive, negative in zip(result["online_positive_slot_indices"], result["online_negative_slot_indices"]):
                    slot_rows.append((tuple(positive.cpu().tolist()), tuple(negative.cpu().tolist())))
        changed = 0 if reference_slots is None else sum(current != initial for current, initial in zip(slot_rows, reference_slots))
        if reference_slots is None:
            reference_slots = slot_rows
        summaries.append({
            "checkpoint": str(checkpoint), "epoch": load_report["joint"]["epoch"],
            "episodes": len(selected), "hard_conflict_slots": conflict_slots,
            "hard_conflict_episodes": conflict_episodes,
            "hard_conflict_episode_rate": conflict_episodes / max(len(selected), 1),
            "hard_conflict_dataset_indices": conflict_dataset_indices,
            "unique_hard_conflict_dataset_indices": sorted(set(conflict_dataset_indices)),
            "hard_slot_changed_episodes_vs_initial": changed,
            "conflict_prompt_nearest_margin": distribution_summary(conflict_margins),
            "soft_overlap": {
                str(temperature): {
                    "overall": distribution_summary(overlaps[temperature]),
                    "hard_conflict": distribution_summary(conflict_overlaps[temperature]),
                    "hard_nonconflict": distribution_summary(nonconflict_overlaps[temperature]),
                } for temperature in overlaps
            },
        })
        del model; torch.cuda.empty_cache()
        print(summaries[-1], flush=True)
    report = {
        "timestamp": stamp, "split": args.split, "test_used": False, "selected_episodes": len(selected),
        "summaries": summaries,
        "reproducibility": {"command": [sys.executable, *sys.argv]},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
