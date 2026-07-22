#!/usr/bin/env python3
"""Create a persistent paired J5/J10 validation bootstrap report."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from benchmarks.v4.phase_6_mask_decoder.src.paired_validation import paired_episode_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-metrics", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase6/evaluation"))
    parser.add_argument("--timestamp")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--noninferiority-margin", type=float, default=0.001)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"paired_validation_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(args.episode_metrics)
    required = {
        f"{model}_pixel_{name}"
        for model in ("baseline", "joint") for name in ("tp", "fp", "fn")
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"episode metrics missing columns: {missing}")
    if len(frame) != 4000:
        raise ValueError("formal paired validation requires exactly 4000 sampled occurrences")
    unique_episodes = int(frame["episode_index"].nunique())
    report = paired_episode_bootstrap(
        frame, samples=int(args.bootstrap_samples), seed=int(args.seed),
        noninferiority_margin=float(args.noninferiority_margin),
    )
    report.update({
        "timestamp": stamp,
        "split": "val",
        "test_used": False,
        "sampled_occurrences": int(len(frame)),
        "unique_episode_indices": unique_episodes,
        "duplicate_sampled_occurrences": int(len(frame) - unique_episodes),
        "inputs": {
            "episode_metrics": str(args.episode_metrics),
            "episode_metrics_sha256": sha256(args.episode_metrics),
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        },
    })
    output.mkdir(parents=True)
    (output / "paired_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
