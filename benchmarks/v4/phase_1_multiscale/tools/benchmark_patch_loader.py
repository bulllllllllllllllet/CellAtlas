#!/usr/bin/env python3
"""Measure dynamic patch-reader throughput using the training Dataset path."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json
from benchmarks.v4.phase_1_multiscale.src.dataset import SegmentationDataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse(); config = load_config(args.config)
    if args.num_samples < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("sample, batch, and worker counts must be positive (workers may be zero)")
    dataset = SegmentationDataset(args.patch_index, config, args.split)
    if args.num_samples > len(dataset):
        raise ValueError("requested more samples than available")
    rng = np.random.default_rng(args.seed)
    subset = Subset(dataset, rng.choice(len(dataset), size=args.num_samples, replace=False).tolist())
    kwargs = {"num_workers": args.num_workers, "pin_memory": True, "persistent_workers": bool(args.num_workers)}
    if args.num_workers:
        kwargs.update({"multiprocessing_context": "spawn", "prefetch_factor": 1})
    loader = DataLoader(subset, batch_size=args.batch_size, **kwargs)
    started = time.monotonic(); count = 0
    for batch in loader:
        count += len(batch["patch_id"])
    elapsed = time.monotonic() - started
    output = create_run_dir(config, "patch_loader_benchmark", args.timestamp)
    save_json(output / "benchmark.json", {"patch_index": str(args.patch_index), "split": args.split, "samples": count, "batch_size": args.batch_size, "num_workers": args.num_workers, "elapsed_seconds": elapsed, "patches_per_second": count / elapsed})
    print(output / "benchmark.json")


if __name__ == "__main__":
    main()
