#!/usr/bin/env python3
"""Create a fixed patient-safe development cohort from the full v4 manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=int, default=140)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def select_train(frame: pd.DataFrame, count: int, class_ids: list[int], rng: np.random.Generator) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"requested {count} train WSI but only {len(frame)} available")
    rows = frame.reset_index(drop=True)
    selected: set[int] = set()
    present = {class_id: [] for class_id in class_ids}
    for index, counts_json in enumerate(rows["class_pixel_counts"]):
        counts = json.loads(counts_json) if isinstance(counts_json, str) else counts_json
        for class_id in class_ids:
            if int(counts.get(str(class_id), 0)):
                present[class_id].append(index)
    missing = [class_id for class_id, indices in present.items() if not indices]
    if missing:
        raise ValueError(f"full train split lacks classes: {missing}")
    # Pick one deterministic carrier per class first, then fill the remaining
    # budget randomly. This preserves rare-class coverage without duplicating WSI.
    for class_id in class_ids:
        selected.add(int(rng.choice(present[class_id])))
    remaining = np.asarray(sorted(set(range(len(rows))) - selected))
    extra = count - len(selected)
    if extra < 0:
        raise ValueError("train cohort size is smaller than required class coverage")
    if extra:
        selected.update(map(int, rng.choice(remaining, size=extra, replace=False)))
    return rows.iloc[sorted(selected)].copy()


def select_split(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"requested {count} WSI but only {len(frame)} available")
    # Dominant-class strata retain broad tissue composition in validation/test.
    groups = [group.index.to_numpy() for _, group in frame.groupby("dominant_class", sort=True)]
    selected: set[int] = set()
    for indices in groups:
        if len(selected) == count:
            break
        selected.add(int(rng.choice(indices)))
    remaining = np.asarray(sorted(set(frame.index) - selected))
    extra = count - len(selected)
    if extra:
        selected.update(map(int, rng.choice(remaining, size=extra, replace=False)))
    return frame.loc[sorted(selected)].copy()


def main() -> None:
    args = parse()
    if min(args.train, args.val, args.test) < 1:
        raise ValueError("all cohort split sizes must be positive")
    config = load_config(args.config)
    manifest = pd.read_parquet(args.manifest)
    rng = np.random.default_rng(args.seed)
    class_ids = [int(item["id"]) for item in config["data"]["class_map"]]
    chosen = [
        select_train(manifest.query("split == 'train'"), args.train, class_ids, rng),
        select_split(manifest.query("split == 'val'"), args.val, rng),
        select_split(manifest.query("split == 'test'"), args.test, rng),
    ]
    cohort = pd.concat(chosen, ignore_index=True).sort_values(["split", "wsi_id"]).reset_index(drop=True)
    if cohort["patient_id"].duplicated().any():
        raise RuntimeError("cohort violates patient isolation")
    output = create_run_dir(config, "cohort", args.timestamp)
    path = output / "cohort_manifest.parquet"
    cohort.to_parquet(path, index=False)
    save_json(output / "cohort_metadata.json", {
        "source_manifest": str(args.manifest), "cohort_manifest": str(path), "seed": args.seed,
        "split_counts": cohort["split"].value_counts().to_dict(), "class_ids": class_ids,
    })
    print(path)


if __name__ == "__main__":
    main()
