#!/usr/bin/env python3
"""Materialize immutable episode geometry from an audited occurrence list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.baseline.common import atomic_json, new_output_directory, sha256_path, timestamp, validate_episode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence-source", type=Path, required=True, help="Parquet with episode_index and frozen occurrence order")
    parser.add_argument("--prompt-geometry", type=Path, required=True, help="Frozen region geometry keyed by episode_index; required for small/large boxes")
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--expected-episodes", type=int, default=4000)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def _json(value) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    output = new_output_directory(args.output_root, "contracts", stamp)
    source = pd.read_parquet(args.occurrence_source).reset_index(drop=True)
    required_source = {"episode_index", "patch_id", "wsi_id", "target_class", "prompt_size"}
    if required_source - set(source):
        raise ValueError(f"occurrence source misses {sorted(required_source - set(source))}")
    if len(source) != args.expected_episodes:
        raise ValueError(f"occurrence count {len(source)} != expected {args.expected_episodes}")
    repeated_occurrences = int(source["episode_index"].duplicated().sum())
    geometry = pd.read_parquet(args.prompt_geometry)
    required_geometry = {
        "episode_index", "positive_points_10x", "negative_points_10x",
        "positive_box_10x", "source_region_ids", "negative_source_region_ids",
    }
    if required_geometry - set(geometry):
        raise ValueError(f"prompt geometry misses {sorted(required_geometry - set(geometry))}")
    if geometry["episode_index"].duplicated().any():
        raise ValueError("prompt geometry has duplicate episode_index")
    geometry = geometry.set_index("episode_index")
    cfg = load_config(args.phase5_config)
    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, args.split, int(cfg["project"]["seed"]),
        cfg["data"]["size_probabilities"], tuple(cfg["data"]["class_ids"]),
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["centroid_knn"]), args.eligibility_index,
    )
    patch_frame = pd.read_parquet(args.patch_index)
    patches = patch_frame[patch_frame["split"] == args.split].set_index("patch_id")
    rows = []
    for order, occurrence in source.iterrows():
        index = int(occurrence.episode_index)
        if index < 0 or index >= len(episodes):
            raise IndexError(f"episode_index {index} outside eligibility dataset")
        item = episodes[index]
        for key in ("patch_id", "wsi_id", "prompt_size"):
            if str(item[key]) != str(occurrence[key]):
                raise RuntimeError(f"episode source mismatch index={index} field={key}")
        if int(item["target_class"]) != int(occurrence.target_class):
            raise RuntimeError(f"target class mismatch at episode_index={index}")
        patch = patches.loc[str(item["patch_id"])]
        geo = geometry.loc[index]
        positive = np.asarray(json.loads(geo.positive_points_10x), dtype=np.float32)
        negative = np.asarray(json.loads(geo.negative_points_10x), dtype=np.float32)
        positive_slots = item["positive_slot_indices"][item["positive_mask"]].numpy().astype(np.int64)
        negative_slots = item["negative_slot_indices"][item["negative_mask"]].numpy().astype(np.int64)
        frozen_positive_slots = np.asarray(json.loads(geo.source_region_ids), dtype=np.int64)
        frozen_negative_slots = np.asarray(json.loads(geo.negative_source_region_ids), dtype=np.int64)
        if not np.array_equal(positive_slots, frozen_positive_slots):
            raise RuntimeError(f"positive source slots changed at episode_index={index}")
        if not np.array_equal(negative_slots, frozen_negative_slots):
            raise RuntimeError(f"negative source slots changed at episode_index={index}")
        if positive.shape != (len(positive_slots), 2) or negative.shape != (len(negative_slots), 2):
            raise RuntimeError(f"frozen point count mismatch at episode_index={index}")
        box = geo.positive_box_10x
        if str(item["prompt_size"]) != "point" and (box is None or (isinstance(box, float) and np.isnan(box))):
            raise ValueError(f"missing frozen box for episode_index={index}")
        rows.append({
            "occurrence_id": f"{args.split}_{order:06d}", "occurrence_order": int(order),
            "episode_index": index, "split": args.split, "patch_id": str(item["patch_id"]),
            "wsi_id": str(item["wsi_id"]), "patient_id": str(patch.patient_id),
            "target_class": int(item["target_class"]), "prompt_size": str(item["prompt_size"]),
            "positive_points_10x": _json(positive), "negative_points_10x": _json(negative),
            "positive_box_10x": None if box is None or (isinstance(box, float) and np.isnan(box)) else _json(json.loads(box) if isinstance(box, str) else box),
            "source_region_ids": _json(json.loads(geo.source_region_ids) if isinstance(geo.source_region_ids, str) else geo.source_region_ids),
            **{name: int(patch[name]) for name in ("x_10x", "y_10x", "width_10x", "height_10x", "x_level0", "y_level0", "width_level0", "height_level0")},
            "wsi_path": str(patch.wsi_path), "gt_path": str(patch.gt_path),
        })
    manifest = pd.DataFrame(rows)
    audit = validate_episode_manifest(manifest, args.split)
    path = output / f"episode_manifest_{stamp}.parquet"
    manifest.to_parquet(path, index=False)
    atomic_json(output / f"contract_metadata_{stamp}.json", {
        "timestamp": stamp, "audit": audit, "manifest": str(path), "manifest_sha256": sha256_path(path),
        "unique_episode_indices": int(source["episode_index"].nunique()),
        "repeated_occurrences": repeated_occurrences,
        "inputs": {name: {"path": str(value), "sha256": sha256_path(value)} for name, value in {
            "occurrence_source": args.occurrence_source, "prompt_geometry": args.prompt_geometry,
            "cache_index": args.cache_index, "label_index": args.label_index,
            "patch_index": args.patch_index, "eligibility_index": args.eligibility_index,
            "phase5_config": args.phase5_config,
        }.items()},
    })
    print(json.dumps({"output": str(output), "manifest": str(path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
