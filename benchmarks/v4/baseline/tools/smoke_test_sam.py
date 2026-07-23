#!/usr/bin/env python3
"""Run and retain one real validation-patch SAM dependency smoke test."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from benchmarks.v4.baseline.adapters.base import EpisodeRequest
from benchmarks.v4.baseline.adapters import build_adapter
from benchmarks.v4.baseline.common import atomic_json, binary_metric_row, new_output_directory, sha256_path, timestamp
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = timestamp(args.timestamp)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["device"] = args.device
    output = new_output_directory(args.output_root, f"{config['name']}_dependency_smoke", stamp)
    result_path = output / f"result_{stamp}.json"
    metadata = {
        "timestamp": stamp,
        "gate": "dependency_smoke_only_not_formal_baseline",
        "split": args.split,
        "test_used": False,
        "episode_index": args.episode_index,
        "device": args.device,
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in {
                "config": args.config,
                "phase2_config": args.phase2_config,
                "phase5_config": args.phase5_config,
                "cache_index": args.cache_index,
                "label_index": args.label_index,
                "patch_index": args.patch_index,
                "eligibility_index": args.eligibility_index,
            }.items()
        },
    }
    shutil.copy2(args.config, output / f"config_snapshot_{stamp}.yaml")
    (output / f"command_{stamp}.txt").write_text(metadata["command"] + "\n", encoding="utf-8")
    try:
        phase2 = load_config(args.phase2_config)
        phase5 = load_config(args.phase5_config)
        dataset = PromptEpisodeDataset(
            args.cache_index,
            args.label_index,
            args.patch_index,
            args.split,
            int(phase5["project"]["seed"]),
            phase5["data"]["size_probabilities"],
            tuple(phase5["data"]["class_ids"]),
            int(phase5["data"]["ignore_index"]),
            int(phase5["data"]["centroid_knn"]),
            args.eligibility_index,
        )
        if not 0 <= args.episode_index < len(dataset):
            raise IndexError(f"episode-index {args.episode_index} outside [0,{len(dataset)})")
        item = dataset[args.episode_index]
        if str(item["prompt_size"]) != "point":
            raise ValueError("SAM dependency smoke requires a point episode; no box geometry may be synthesized")
        patches = pd.read_parquet(args.patch_index)
        matches = patches[patches["patch_id"].astype(str) == str(item["patch_id"])]
        if len(matches) != 1 or str(matches.iloc[0]["split"]) != args.split:
            raise RuntimeError("episode patch identity/split mismatch")
        patch = matches.iloc[0]
        image = read_he_patch(
            Path(patch.wsi_path),
            int(patch.x_level0), int(patch.y_level0),
            int(patch.width_level0), int(patch.height_level0), int(patch.width_10x),
        )
        gt = decode_gt_patch(
            Path(patch.gt_path),
            int(patch.x_10x), int(patch.y_10x),
            int(patch.width_10x), int(patch.height_10x),
            phase2["data"]["class_map"], int(phase2["data"]["ignore_index"]),
        )
        size = np.asarray([patch.width_10x, patch.height_10x], dtype=np.float32)
        positive = item["positive_xy"][item["positive_mask"]].numpy() * size
        negative = item["negative_xy"][item["negative_mask"]].numpy() * size
        adapter = build_adapter(config)
        request = EpisodeRequest(
            occurrence_id=f"{args.split}_dependency_smoke_{args.episode_index}",
            image=np.asarray(image, dtype=np.uint8),
            positive_points=positive.astype(np.float32),
            negative_points=negative.astype(np.float32),
            prompt_type="point",
            multiscale_inputs={
                "fine": np.asarray(image, dtype=np.uint8),
                "wsi_path": str(patch.wsi_path),
                "center_level0": [
                    int(patch.x_level0) + int(patch.width_level0) / 2,
                    int(patch.y_level0) + int(patch.height_level0) / 2,
                ],
                "fine_box_level0": [
                    int(patch.x_level0), int(patch.y_level0),
                    int(patch.width_level0), int(patch.height_level0),
                ],
            } if bool(config.get("requires_multiscale_inputs", False)) else None,
        )
        prediction = adapter.timed_predict(request)
        prediction.validate(gt.shape)
        target_class = int(item["target_class"])
        valid = gt != int(phase2["data"]["ignore_index"])
        truth = gt == target_class
        metrics = binary_metric_row(prediction.binary_mask, truth, valid, int(config["boundary_tolerance"]))
        arrays_path = output / f"prediction_arrays_{stamp}.npz"
        np.savez_compressed(
            arrays_path,
            probability=prediction.probability.astype(np.float32),
            binary_mask=prediction.binary_mask,
            target_mask=truth,
            valid_mask=valid,
            positive_points=positive.astype(np.float32),
            negative_points=negative.astype(np.float32),
        )
        metadata |= {
            "status": "completed",
            "patch_id": str(item["patch_id"]),
            "wsi_id": str(item["wsi_id"]),
            "target_class": target_class,
            "image_shape": list(image.shape),
            "positive_points": positive.tolist(),
            "negative_points": negative.tolist(),
            "latency_ms": prediction.latency_ms,
            "peak_memory_mb": prediction.peak_memory_mb,
            "probability_range": [float(prediction.probability.min()), float(prediction.probability.max())],
            "candidates": prediction.candidates,
            "adapter_provenance": adapter.provenance(),
            "metrics": metrics,
            "arrays": str(arrays_path),
            "arrays_sha256": sha256_path(arrays_path),
        }
        atomic_json(result_path, metadata)
    except Exception as exc:
        metadata |= {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result_path, metadata)
        raise
    print(json.dumps({"output": str(output), "result": str(result_path), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
