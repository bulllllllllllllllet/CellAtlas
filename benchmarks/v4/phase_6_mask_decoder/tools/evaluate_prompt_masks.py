#!/usr/bin/env python3
"""Evaluate prompt models with macro, prompt-excluded, and pixel metrics on validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import load_phase2
from benchmarks.v4.phase_4_cross_scale.tools.build_fine_region_labels import LabelSourceDataset
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import (
    EpisodeBalancedSampler,
    PromptEpisodeDataset,
    collate_prompt_episodes,
)
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import (
    binary_counts,
    boundary_f1,
    dice_from_counts,
    summarize_episode_rows,
)
from benchmarks.v4.phase_6_mask_decoder.src.model import (
    ContextAwareMaskDecoder,
    load_phase5_prompt_model,
    project_region_probabilities,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--phase6-checkpoint", type=Path)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase6/evaluation"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    files = [
        Path(__file__),
        Path("benchmarks/v4/phase_5_prompt_encoder/src/dataset.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/evaluation.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/model.py"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


class PixelEpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: PromptEpisodeDataset,
        episode_indices: list[int],
        patch_index: Path,
        class_map: list[dict],
        ignore_index: int,
    ):
        self.episodes = episodes
        self.indices = list(map(int, episode_indices))
        patches = pd.read_parquet(patch_index).query("split == 'val'").set_index("patch_id")
        required = set(episodes.patch_ids[self.indices])
        missing = sorted(required - set(patches.index))
        if missing:
            raise ValueError(f"selected episodes have {len(missing)} patches absent from patch index")
        rows = []
        for patch_id in episodes.patch_ids[self.indices]:
            row = patches.loc[str(patch_id)].to_dict()
            row.update({"patch_id": str(patch_id), "split": "val"})
            rows.append(row)
        self.pixel_source = LabelSourceDataset(rows, class_map, int(ignore_index))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[self.indices[index]]
        pixel = self.pixel_source[index]
        if pixel["patch_id"] != episode["patch_id"]:
            raise RuntimeError("episode/pixel patch ordering mismatch")
        episode["image"] = pixel["image"]
        episode["pixel_gt"] = pixel["mask"]
        return episode


def move_batch(batch: dict, device: torch.device) -> dict:
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def load_phase6(checkpoint_path: Path, phase5_checkpoint: Path, device: torch.device) -> ContextAwareMaskDecoder:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]; model_cfg = cfg["model"]
    prompt, _ = load_phase5_prompt_model(
        str(phase5_checkpoint), int(model_cfg["region_dim"]), int(model_cfg["prompt_heads"]),
        int(model_cfg["prompt_set_layers"]), float(model_cfg["prompt_dropout"]),
    )
    model = ContextAwareMaskDecoder(
        prompt, int(model_cfg["region_dim"]), int(model_cfg["graph_heads"]),
        int(model_cfg["graph_layers"]), int(model_cfg["graph_neighbours"]),
        float(model_cfg["graph_dropout"]), bool(model_cfg["freeze_prompt_encoder"]),
        float(model_cfg.get("residual_limit", 1.0)),
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Phase6 load mismatch missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


def count_row(prefix: str, counts: dict[str, torch.Tensor], position: int, row: dict) -> None:
    for name in ("tp", "fp", "fn", "tn", "positive", "valid"):
        row[f"{prefix}_{name}"] = int(counts[name][position].item())
    row[f"{prefix}_dice"] = float(
        dice_from_counts(row[f"{prefix}_tp"], row[f"{prefix}_fp"], row[f"{prefix}_fn"])
    )


def grouped_macro(frame: pd.DataFrame, model_names: tuple[str, ...]) -> dict:
    result = {}
    for column in ("target_class", "prompt_size"):
        groups = {}
        for value, part in frame.groupby(column, sort=True):
            groups[str(value)] = {
                model: {
                    scope: float(part[f"{model}_{scope}_dice"].mean(skipna=True))
                    for scope in ("region", "unprompted_region", "pixel")
                }
                for model in model_names
            }
        result[f"by_{column}"] = groups
    return result


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("pixel projection evaluation requires CUDA")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"prompt_mask_metrics_{stamp}"
    exists = torch.tensor(int(output.exists()), device=device)
    if world > 1:
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if bool(exists.item()):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True); (output / "shards").mkdir()
    if world > 1:
        dist.barrier()

    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, "val",
        seed=int(p5cfg["project"]["seed"]), size_probabilities=p5cfg["data"]["size_probabilities"],
        target_class_ids=tuple(p5cfg["data"]["class_ids"]), ignore_index=int(p5cfg["data"]["ignore_index"]),
        centroid_knn=int(p5cfg["data"]["centroid_knn"]), eligibility_index=args.eligibility_index,
    )
    sampler = EpisodeBalancedSampler(
        episodes, p5cfg["data"]["size_probabilities"], p5cfg["data"]["group_probabilities"],
        tuple(p5cfg["data"]["class_ids"]), int(p5cfg["project"]["seed"]) + 10_000_019,
        epoch_size=min(int(args.validation_episodes), len(episodes)),
    )
    selected = list(iter(sampler)); local_indices = selected[rank::world]
    dataset = PixelEpisodeDataset(
        episodes, local_indices, args.patch_index, p2cfg["data"]["class_map"], int(p2cfg["data"]["ignore_index"]),
    )
    loader_kwargs = dict(
        batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers),
        pin_memory=True, collate_fn=collate_prompt_episodes,
    )
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)

    phase2 = load_phase2(p2cfg, args.phase2_checkpoint, device)
    p5_model_cfg = p5cfg["model"]
    phase5, phase5_transfer = load_phase5_prompt_model(
        str(args.phase5_checkpoint), int(p5_model_cfg["region_dim"]), int(p5_model_cfg["heads"]),
        int(p5_model_cfg["set_layers"]), float(p5_model_cfg["dropout"]),
    )
    phase5 = phase5.to(device).eval()
    phase6 = load_phase6(args.phase6_checkpoint, args.phase5_checkpoint, device) if args.phase6_checkpoint else None
    model_names = ("phase5", "phase6") if phase6 is not None else ("phase5",)
    ignore = int(p2cfg["data"]["ignore_index"]); classes = len(p2cfg["data"]["class_map"])
    amp_dtype = torch.bfloat16; rows = []; mismatch_slots = 0; compared_slots = 0; max_initial_difference = 0.0
    t0 = time.monotonic(); processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=amp_dtype):
                phase2_output = phase2(batch["image"], return_full_assignment=True, return_tokens=False)
                p5_output = phase5(batch)
                p6_output = phase6(batch) if phase6 is not None else None
            recomputed_labels = region_labels(phase2_output["assignment_low"], batch["pixel_gt"], classes, ignore)
            recomputed_binary = torch.full_like(recomputed_labels, ignore)
            valid_recomputed = (recomputed_labels != ignore) & batch["fine_active"]
            recomputed_binary[valid_recomputed] = (recomputed_labels[valid_recomputed] == batch["target_class"][:, None].expand_as(recomputed_labels)[valid_recomputed]).long()
            comparison = batch["fine_active"] | (batch["binary_target"] != ignore)
            mismatch_slots += int(((recomputed_binary != batch["binary_target"]) & comparison).sum().item())
            compared_slots += int(comparison.sum().item())

            outputs = {"phase5": p5_output["logits"]}
            if p6_output is not None:
                difference = (p6_output["initial_logits"] - p5_output["logits"]).abs().max()
                max_initial_difference = max(max_initial_difference, float(difference))
                outputs["phase6"] = p6_output["logits"]
            region_truth = batch["binary_target"] == 1; region_valid = batch["binary_target"] != ignore
            unprompted_valid = region_valid & ~batch["all_prompted_regions"]
            pixel_valid = batch["pixel_gt"] != ignore
            pixel_truth = batch["pixel_gt"] == batch["target_class"][:, None, None]

            computed = {}
            for name, logits in outputs.items():
                region_prediction = logits >= 0
                region = binary_counts(region_prediction, region_truth, region_valid)
                query = binary_counts(region_prediction, region_truth, unprompted_valid)
                projected = project_region_probabilities(phase2_output["assignment"].float(), logits.float())
                pixel_prediction = projected["pixel_probability"] >= 0.5
                pixel = binary_counts(pixel_prediction, pixel_truth, pixel_valid)
                boundary = boundary_f1(pixel_prediction, pixel_truth, pixel_valid, int(args.boundary_tolerance))
                computed[name] = (region, query, pixel, boundary)
            for position, patch_id in enumerate(batch["patch_id"]):
                row = {
                    "patch_id": patch_id, "wsi_id": batch["wsi_id"][position],
                    "target_class": int(batch["target_class"][position]), "prompt_size": batch["prompt_size"][position],
                    "rank": rank,
                }
                for name in model_names:
                    region, query, pixel, boundary = computed[name]
                    count_row(f"{name}_region", region, position, row)
                    count_row(f"{name}_unprompted_region", query, position, row)
                    count_row(f"{name}_pixel", pixel, position, row)
                    row[f"{name}_boundary_f1"] = float(boundary["boundary_f1"][position])
                    row[f"{name}_boundary_evaluable"] = bool(boundary["boundary_evaluable"][position])
                rows.append(row)
            processed += len(batch["patch_id"])
            if processed % 100 < len(batch["patch_id"]):
                print({"event": "evaluation_progress", "rank": rank, "processed": processed, "total": len(dataset)}, flush=True)
    if mismatch_slots:
        raise RuntimeError(f"recomputed/cached region labels disagree in {mismatch_slots}/{compared_slots} compared slots")
    if max_initial_difference > 1e-5:
        raise RuntimeError(f"Phase6 embedded Phase5 differs from source model: {max_initial_difference}")
    shard = output / "shards" / f"episodes_rank{rank:02d}.parquet"
    pd.DataFrame(rows).to_parquet(shard, index=False)
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = sorted((output / "shards").glob("episodes_rank*.parquet"))
        frame = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
        if len(frame) != len(selected):
            raise RuntimeError(f"merged episode count {len(frame)} != selected {len(selected)}")
        frame.to_parquet(output / "episode_metrics.parquet", index=False)
        summary = summarize_episode_rows(frame.to_dict("records"), model_names)
        summary.update(grouped_macro(frame, model_names))
        summary.update({
            "timestamp": stamp, "split": "val", "test_used": False, "world_size": world,
            "boundary_tolerance_pixels": int(args.boundary_tolerance), "region_label_mismatch_slots": mismatch_slots,
            "phase5_phase6_initial_max_abs_difference": max_initial_difference,
            "elapsed_seconds": time.monotonic() - t0,
            "inputs": {name: str(getattr(args, name)) if getattr(args, name) is not None else None for name in (
                "phase2_config", "phase2_checkpoint", "phase5_config", "phase5_checkpoint", "phase6_checkpoint",
                "cache_index", "label_index", "patch_index", "eligibility_index",
            )},
            "phase5_transfer": phase5_transfer,
            "reproducibility": {"command": [sys.executable, *sys.argv], "source_sha256": source_hashes()},
        })
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
