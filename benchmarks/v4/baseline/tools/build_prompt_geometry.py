#!/usr/bin/env python3
"""Freeze prompt points/boxes from Phase-2 hard assignments without consulting GT."""
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


def hard_region_representatives(assignment: np.ndarray, slots: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Choose hard-member maxima, using the slot-probability maximum for empty slots."""
    assignment = np.asarray(assignment)
    slots = np.asarray(slots, dtype=np.int64)
    if assignment.ndim != 3:
        raise ValueError(f"assignment must be [slots,H,W], got {assignment.shape}")
    if slots.ndim != 1 or not len(slots) or len(np.unique(slots)) != len(slots):
        raise ValueError("slots must be a non-empty unique 1D array")
    if slots.min() < 0 or slots.max() >= assignment.shape[0]:
        raise ValueError(f"slot indices must be in [0,{assignment.shape[0]})")
    hard = assignment.argmax(0)
    points = []
    modes = []
    for slot in slots:
        yy, xx = np.nonzero(hard == int(slot))
        if not len(xx):
            y, x = np.unravel_index(int(np.argmax(assignment[int(slot)])), hard.shape)
            points.append([float(x) + 0.5, float(y) + 0.5])
            modes.append("soft_max_probability_empty_hard_slot")
            continue
        probability = assignment[int(slot), yy, xx]
        selected = int(np.argmax(probability))
        points.append([float(xx[selected]) + 0.5, float(yy[selected]) + 0.5])
        modes.append("hard_member_max_probability")
    return np.asarray(points, dtype=np.float32), modes


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
    parser.add_argument("--episode-index", type=int, help="Export one explicit audited episode index for a representative run")
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
    if args.episode_index is not None:
        occurrences = occurrences[occurrences["episode_index"] == args.episode_index].reset_index(drop=True)
        if len(occurrences) != 1:
            raise ValueError(f"episode_index {args.episode_index} is not present exactly once after de-duplication")
    if args.limit is not None:
        if args.episode_index is not None:
            raise ValueError("--limit and --episode-index are mutually exclusive")
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
        assignment = np.asarray(encoded["assignment"])
        hard = assignment.argmax(0)
        positive_slots = item["positive_slot_indices"][item["positive_mask"]].numpy().astype(np.int64)
        negative_slots = item["negative_slot_indices"][item["negative_mask"]].numpy().astype(np.int64)
        try:
            positive_points, positive_point_modes = hard_region_representatives(assignment, positive_slots)
            negative_points, negative_point_modes = hard_region_representatives(assignment, negative_slots)
        except ValueError as exc:
            append_jsonl(
                failures,
                {"episode_index": index, "patch_id": item["patch_id"], "error": str(exc)},
            )
            flush()
            raise RuntimeError(f"representative-point failure for episode_index={index}") from exc
        selected = np.isin(hard, positive_slots)
        if not selected.any():
            xx = np.floor(positive_points[:, 0]).astype(np.int64)
            yy = np.floor(positive_points[:, 1]).astype(np.int64)
            box_mode = "soft_point_envelope_all_positive_hard_slots_empty"
        else:
            yy, xx = np.nonzero(selected)
            box_mode = "union_hard_assignment_positive_slots"
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        rows.append({
            "episode_index": index, "patch_id": str(item["patch_id"]),
            "positive_points_10x": json.dumps(positive_points.tolist(), separators=(",", ":")),
            "negative_points_10x": json.dumps(negative_points.tolist(), separators=(",", ":")),
            "positive_point_modes": json.dumps(positive_point_modes, separators=(",", ":")),
            "negative_point_modes": json.dumps(negative_point_modes, separators=(",", ":")),
            "positive_box_mode": box_mode,
            "positive_box_10x": json.dumps(box, separators=(",", ":")),
            "source_region_ids": json.dumps(positive_slots.tolist(), separators=(",", ":")),
            "negative_source_region_ids": json.dumps(negative_slots.tolist(), separators=(",", ":")),
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
        "box_rule": "union_bbox_of_positive_hard_assignment_slots_or_soft_point_envelope_when_all_are_empty_no_gt",
        "point_rule": "max_slot_probability_among_hard_assignment_member_pixel_centers_or_soft_max_for_empty_slots_no_gt",
        "empty_hard_slot_fallback": "user_approved_soft_max_slot_probability_pixel_center",
    })
    print(json.dumps({"output": str(output), "geometry": str(geometry_path), "rows": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
