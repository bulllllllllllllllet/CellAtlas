#!/usr/bin/env python3
"""Evaluate one strict adapter shard against an immutable episode manifest."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch
from benchmarks.v4.baseline.adapters import EpisodeRequest, build_adapter
from benchmarks.v4.baseline.common import (
    append_jsonl, atomic_json, binary_metric_row, new_output_directory, parse_json_array,
    sha256_path, timestamp, validate_episode_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True, help="Authoritative class_map/ignore_index")
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, help="Existing shared timestamp directory for rank>0")
    return parser.parse_args()


def environment_record() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def source_hashes() -> dict[str, str]:
    root = Path("benchmarks/v4/baseline")
    return {str(path): sha256_path(path) for path in sorted(root.rglob("*.py"))}


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if args.batch_size < 1 or args.num_workers < 0 or args.shard_size < 1:
        raise ValueError("invalid execution parameters")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "adapter" not in config or "name" not in config:
        raise ValueError("baseline config requires name and adapter")
    if config.get("failure_metric_policy") != "empty_mask":
        raise ValueError("failure_metric_policy must be explicitly frozen to 'empty_mask'")
    phase2 = load_config(args.phase2_config)
    frame = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
    audit = validate_episode_manifest(frame, args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        frame = frame.iloc[: args.limit].copy()
    selected = frame.iloc[args.rank::args.world_size].reset_index(drop=True)
    if args.output_dir is None:
        if args.rank != 0:
            raise ValueError("rank>0 requires --output-dir created by rank 0")
        output = new_output_directory(args.output_root, str(config["name"]), stamp)
    else:
        output = args.output_dir
        if not output.is_dir() or not output.name.endswith(stamp):
            raise ValueError("output-dir must be an existing directory ending in the run timestamp")
    shards = output / f"shards_{stamp}"
    shards.mkdir(exist_ok=True)
    completed_path = output / f"completed_{stamp}.jsonl"
    failures_path = output / f"failures_{stamp}.jsonl"
    completed_path.open("a", encoding="utf-8").close()
    failures_path.open("a", encoding="utf-8").close()
    rank_metadata = output / f"run_metadata_rank{args.rank:02d}_{stamp}.json"
    command_path = output / f"command_rank{args.rank:02d}_{stamp}.txt"
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    if args.rank == 0:
        shutil.copy2(args.config, output / f"config_snapshot_{stamp}.yaml")
        (output / f"environment_{stamp}.json").write_text(json.dumps(environment_record(), indent=2), encoding="utf-8")
    metadata = {
        "timestamp": stamp, "method": config["name"], "split": args.split,
        "test_used": args.split == "test", "rank": args.rank, "world_size": args.world_size,
        "gpus": args.gpus, "num_workers": args.num_workers, "batch_size": args.batch_size,
        "seed": args.seed, "shard_size": args.shard_size, "selected_rows": len(selected),
        "manifest": str(args.episode_manifest), "manifest_sha256": sha256_path(args.episode_manifest),
        "config": str(args.config), "config_sha256": sha256_path(args.config),
        "manifest_audit": audit, "failure_metric_policy": "empty_mask",
        "source_hashes": source_hashes(),
    }
    try:
        adapter = build_adapter(config)
        metadata["adapter_provenance"] = adapter.provenance()
        metadata["gate_status"] = "adapter_loaded"
    except Exception as exc:
        metadata |= {"gate_status": "failed_gate", "gate_error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(rank_metadata, metadata)
        append_jsonl(failures_path, {"event": "failed_gate", "rank": args.rank, "error": repr(exc)})
        raise
    atomic_json(rank_metadata, metadata)

    class_map = phase2["data"]["class_map"]
    ignore = int(phase2["data"]["ignore_index"])
    tolerance = int(config.get("boundary_tolerance", 2))
    rows: list[dict] = []
    shard_index = 0
    run_started = time.perf_counter()

    def flush() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        path = shards / f"episodes_rank{args.rank:02d}_shard{shard_index:05d}_{stamp}.parquet"
        if path.exists():
            raise FileExistsError(path)
        pd.DataFrame(rows).to_parquet(path, index=False)
        rows = []; shard_index += 1

    for record in selected.to_dict("records"):
        started = time.perf_counter(); occurrence_id = str(record["occurrence_id"])
        base = {
            "occurrence_id": occurrence_id, "occurrence_order": int(record["occurrence_order"]),
            "episode_index": int(record["episode_index"]), "patch_id": str(record["patch_id"]),
            "wsi_id": str(record["wsi_id"]), "patient_id": str(record["patient_id"]),
            "target_class": int(record["target_class"]), "prompt_size": str(record["prompt_size"]),
            "rank": args.rank,
        }
        try:
            image = read_he_patch(
                Path(record["wsi_path"]), int(record["x_level0"]), int(record["y_level0"]),
                int(record["width_level0"]), int(record["height_level0"]), int(record["width_10x"]),
            )
            gt = decode_gt_patch(
                Path(record["gt_path"]), int(record["x_10x"]), int(record["y_10x"]),
                int(record["width_10x"]), int(record["height_10x"]), class_map, ignore,
            )
            positive_box = None
            if record.get("positive_box_10x") is not None and not pd.isna(record.get("positive_box_10x")):
                positive_box = np.asarray(json.loads(record["positive_box_10x"]), dtype=np.float32)
            request = EpisodeRequest(
                occurrence_id=occurrence_id, image=np.asarray(image, dtype=np.uint8),
                positive_points=parse_json_array(record["positive_points_10x"], 2),
                negative_points=parse_json_array(record["negative_points_10x"], 2),
                prompt_type=str(record["prompt_size"]), positive_box=positive_box,
                multiscale_inputs={
                    "fine": np.asarray(image, dtype=np.uint8),
                    "wsi_path": str(record["wsi_path"]),
                    "center_level0": [
                        int(record["x_level0"]) + int(record["width_level0"]) / 2,
                        int(record["y_level0"]) + int(record["height_level0"]) / 2,
                    ],
                    "fine_box_level0": [int(record[name]) for name in ("x_level0", "y_level0", "width_level0", "height_level0")],
                } if bool(config.get("requires_multiscale_inputs", False)) else None,
            )
            try:
                prediction = adapter.timed_predict(request)
                status = prediction.status
                mask = prediction.binary_mask if status == "completed" else np.zeros(gt.shape, dtype=bool)
                candidate_info = prediction.candidates
                latency = prediction.latency_ms; peak = prediction.peak_memory_mb
                if status == "abstained":
                    append_jsonl(failures_path, {"occurrence_id": occurrence_id, "status": status})
                else:
                    append_jsonl(completed_path, {"occurrence_id": occurrence_id, "status": status})
            except Exception as exc:
                status = "failed"; mask = np.zeros(gt.shape, dtype=bool); candidate_info = {}
                latency = (time.perf_counter() - started) * 1000.0; peak = 0.0
                append_jsonl(failures_path, {
                    "occurrence_id": occurrence_id, "status": status, "stage": "predict",
                    "error": repr(exc), "traceback": traceback.format_exc(),
                })
            truth = gt == int(record["target_class"]); valid = gt != ignore
            metric = binary_metric_row(mask, truth, valid, tolerance)
            rows.append(base | metric | {
                "status": status, "latency_ms": float(latency), "peak_memory_mb": float(peak),
                "candidate_info": json.dumps(candidate_info, separators=(",", ":"), default=str),
            })
        except Exception as exc:
            append_jsonl(failures_path, {
                "occurrence_id": occurrence_id, "status": "failed_gate", "stage": "input_or_gt",
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
            flush()
            metadata |= {"gate_status": "failed_gate", "failed_occurrence": occurrence_id, "gate_error": repr(exc)}
            atomic_json(rank_metadata, metadata)
            raise
        if len(rows) >= args.shard_size:
            flush()
    flush()
    metadata |= {"gate_status": "completed", "elapsed_seconds": time.perf_counter() - run_started, "shards": shard_index}
    atomic_json(rank_metadata, metadata)
    print(json.dumps({"output": str(output), "rank": args.rank, "episodes": len(selected), "shards": shard_index}, indent=2))


if __name__ == "__main__":
    main()
