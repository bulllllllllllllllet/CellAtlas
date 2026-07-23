#!/usr/bin/env python3
"""Strictly merge one method's shards and compute the frozen metric summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.v4.baseline.common import atomic_json, sha256_path, summarize_metrics, timestamp, validate_episode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--expected-world-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    manifest = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
    validate_episode_manifest(manifest)
    paths = sorted((args.run_dir / f"shards_{stamp}").glob(f"episodes_rank*_shard*_{stamp}.parquet"))
    if not paths:
        raise FileNotFoundError("no matching episode shards")
    ranks = {int(path.name.split("rank", 1)[1].split("_", 1)[0]) for path in paths}
    if ranks != set(range(args.expected_world_size)):
        raise ValueError(f"rank coverage {sorted(ranks)} != expected {list(range(args.expected_world_size))}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if frame["occurrence_id"].duplicated().any():
        raise ValueError("merged shards contain duplicate occurrence_id")
    frame = frame.sort_values("occurrence_order").reset_index(drop=True)
    expected = manifest["occurrence_id"].astype(str).tolist()
    actual = frame["occurrence_id"].astype(str).tolist()
    if actual != expected:
        missing = sorted(set(expected) - set(actual)); extra = sorted(set(actual) - set(expected))
        raise ValueError(f"occurrence mismatch missing={missing[:10]} extra={extra[:10]}")
    identity = ["occurrence_id", "occurrence_order", "episode_index", "patch_id", "wsi_id", "patient_id", "target_class", "prompt_size"]
    if not frame[identity].astype(str).equals(manifest[identity].astype(str)):
        raise ValueError("merged metric identities differ from the episode manifest")
    summary = summarize_metrics(frame)
    summary |= {
        "timestamp": stamp, "manifest": str(args.episode_manifest),
        "manifest_sha256": sha256_path(args.episode_manifest), "shard_count": len(paths),
        "formal_main_table_eligible": bool(summary["failed"] == 0 and summary["abstained"] == 0 and summary["coverage"] == 1.0),
    }
    metrics_path = args.run_dir / f"episode_metrics_{stamp}.parquet"
    summary_path = args.run_dir / f"summary_{stamp}.json"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError("merged output already exists")
    frame.to_parquet(metrics_path, index=False)
    atomic_json(summary_path, summary)
    print(json.dumps({"episode_metrics": str(metrics_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
