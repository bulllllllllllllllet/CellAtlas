#!/usr/bin/env python3
"""Freeze an N-positive/one-negative point-only evaluation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.v4.baseline.common import atomic_json, new_output_directory, sha256_path, timestamp, validate_episode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--prompt-geometry", type=Path, required=True,
                        help="Frozen geometry carrying negative hard-assignment source slots.")
    parser.add_argument("--positive-points", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    source = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
    geometry = pd.read_parquet(args.prompt_geometry).set_index("episode_index")
    keep = []
    for row in source.itertuples(index=False):
        positive = json.loads(row.positive_points_10x); negative = json.loads(row.negative_points_10x)
        positive_slots = json.loads(row.source_region_ids)
        if int(row.episode_index) not in geometry.index:
            raise KeyError(f"frozen geometry misses episode_index={row.episode_index}")
        geo = geometry.loc[int(row.episode_index)]
        if geo.patch_id != row.patch_id or json.loads(geo.negative_points_10x) != negative:
            raise ValueError(f"frozen geometry mismatch for episode_index={row.episode_index}")
        negative_slots = json.loads(geo.negative_source_region_ids)
        if len(positive) >= args.positive_points and len(negative) >= 1:
            item = row._asdict()
            item["positive_points_10x"] = json.dumps(positive[:args.positive_points], separators=(",", ":"))
            item["negative_points_10x"] = json.dumps(negative[:1], separators=(",", ":"))
            item["source_region_ids"] = json.dumps(positive_slots[:args.positive_points], separators=(",", ":"))
            item["negative_source_region_ids"] = json.dumps(negative_slots[:1], separators=(",", ":"))
            item["positive_box_10x"] = None; item["prompt_size"] = "point"
            keep.append(item)
    frame = pd.DataFrame(keep)
    frame["occurrence_order"] = range(len(frame))
    frame["occurrence_id"] = [f"{value}_p{args.positive_points}n1" for value in frame["occurrence_id"]]
    audit = validate_episode_manifest(frame, args.split)
    output = new_output_directory(args.output_root, f"point_{args.positive_points}p_1n_manifest", stamp)
    path = output / f"episode_manifest_{stamp}.parquet"; frame.to_parquet(path, index=False)
    atomic_json(output / f"metadata_{stamp}.json", {
        "timestamp": stamp, "source": str(args.episode_manifest), "source_sha256": sha256_path(args.episode_manifest),
        "prompt_geometry": str(args.prompt_geometry), "prompt_geometry_sha256": sha256_path(args.prompt_geometry),
        "manifest": str(path), "manifest_sha256": sha256_path(path), "audit": audit,
        "rule": f"first_{args.positive_points}_frozen_positive_and_first_frozen_negative_point_no_box",
    })
    print(json.dumps({"manifest": str(path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
