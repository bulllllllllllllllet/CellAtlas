#!/usr/bin/env python3
"""Freeze prompt boxes from Phase-2 assignments without consulting GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import encode_scale, load_phase2
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.baseline.common import append_jsonl, atomic_json, new_output_directory, sha256_path, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence-source", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    if args.batch_size != 1:
        raise ValueError("geometry exporter currently requires --batch-size 1")
    if args.num_workers != 0:
        raise ValueError("geometry exporter performs direct deterministic reads; --num-workers must be 0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    model = load_phase2(p2cfg, args.phase2_checkpoint, device)
    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, args.split, int(p5cfg["project"]["seed"]),
        p5cfg["data"]["size_probabilities"], tuple(p5cfg["data"]["class_ids"]),
        int(p5cfg["data"]["ignore_index"]), int(p5cfg["data"]["centroid_knn"]), args.eligibility_index,
    )
    occurrence_rows = pd.read_parquet(args.occurrence_source).reset_index(drop=True)
    occurrences = occurrence_rows.drop_duplicates("episode_index", keep="first").reset_index(drop=True)
    if args.limit is not None:
        occurrences = occurrences.iloc[: args.limit]
    patches = pd.read_parquet(args.patch_index)
    patches = patches[patches["split"] == args.split].set_index("patch_id")
    output = new_output_directory(args.output_root, "prompt_geometry", stamp)
    shard_dir = output / f"shards_{stamp}"; shard_dir.mkdir()
    completed = output / f"completed_{stamp}.jsonl"; failures = output / f"failures_{stamp}.jsonl"
    completed.open("x", encoding="utf-8").close(); failures.open("x", encoding="utf-8").close()
    rows = []; shard_index = 0

    def flush() -> None:
        nonlocal rows, shard_index
        if not rows: return
        path = shard_dir / f"geometry_shard{shard_index:05d}_{stamp}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False); rows = []; shard_index += 1

    for occurrence in occurrences.itertuples(index=False):
        index = int(occurrence.episode_index); item = episodes[index]
        if str(item["patch_id"]) != str(occurrence.patch_id):
            raise RuntimeError(f"episode/occurrence mismatch at index={index}")
        patch = patches.loc[str(item["patch_id"])]
        image = read_he_patch(
            Path(patch.wsi_path), int(patch.x_level0), int(patch.y_level0),
            int(patch.width_level0), int(patch.height_level0), int(patch.width_10x),
        )
        encoded = encode_scale(model, image, device, torch.bfloat16)
        hard = np.asarray(encoded["assignment"]).argmax(0)
        slots = item["positive_slot_indices"][item["positive_mask"]].numpy().astype(np.int64)
        selected = np.isin(hard, slots)
        if not selected.any():
            append_jsonl(failures, {"episode_index": index, "patch_id": item["patch_id"], "error": "positive slots have no assigned pixels"})
            flush(); raise RuntimeError(f"positive slots have no pixels for episode_index={index}")
        yy, xx = np.nonzero(selected)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        rows.append({
            "episode_index": index, "patch_id": str(item["patch_id"]),
            "positive_box_10x": json.dumps(box, separators=(",", ":")),
            "source_region_ids": json.dumps(slots.tolist(), separators=(",", ":")),
            "assignment_checkpoint_sha256": sha256_path(args.phase2_checkpoint),
        })
        append_jsonl(completed, {"episode_index": index, "patch_id": item["patch_id"]})
        if len(rows) >= args.shard_size: flush()
    flush()
    paths = sorted(shard_dir.glob(f"geometry_shard*_{stamp}.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if len(frame) != len(occurrences) or frame["episode_index"].duplicated().any():
        raise RuntimeError("geometry shard merge invariant failed")
    geometry_path = output / f"prompt_geometry_{stamp}.parquet"; frame.to_parquet(geometry_path, index=False)
    atomic_json(output / f"metadata_{stamp}.json", {
        "timestamp": stamp, "split": args.split, "test_used": args.split == "test",
        "rows": len(frame), "source_occurrences": len(occurrence_rows),
        "repeated_occurrences": int(occurrence_rows["episode_index"].duplicated().sum()),
        "gpus": args.gpus, "device": str(device), "seed": args.seed,
        "phase2_checkpoint": str(args.phase2_checkpoint), "phase2_checkpoint_sha256": sha256_path(args.phase2_checkpoint),
        "geometry": str(geometry_path), "geometry_sha256": sha256_path(geometry_path),
        "rule": "union_bbox_of_frozen_positive_phase2_assignment_slots_no_gt",
    })
    print(json.dumps({"output": str(output), "geometry": str(geometry_path), "rows": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
