#!/usr/bin/env python3
"""Build an auditable manifest of feasible cached Phase-5 prompt episodes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.phase_5_prompt_encoder.src.prompts import (
    PROMPT_SIZE_SPECS,
    centroid_knn_adjacency,
    sample_connected_region_set,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--class-ids", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--centroid-knn", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--limit-per-split", type=int)
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"prompt_eligibility_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    cache = pd.read_parquet(args.cache_index)
    cache = cache[cache["split"].isin(args.splits)].sort_values(["split", "patch_id"])
    labels = pd.read_parquet(args.label_index).set_index("patch_id")
    missing = cache.loc[~cache["patch_id"].isin(labels.index), "patch_id"]
    if len(missing):
        raise ValueError(f"label index missing {len(missing)} cache patches")

    rows: list[dict] = []
    completed = output / "completed.jsonl"
    episode_records = output / "eligible_episodes.jsonl"
    counts = {}
    for split in args.splits:
        part = cache[cache["split"] == split]
        if args.limit_per_split is not None:
            part = part.iloc[: args.limit_per_split]
        for row in part.to_dict("records"):
            patch_episode_rows: list[dict] = []
            with np.load(row["shard_path"]) as archive:
                active = archive["fine_active"].astype(bool)
                cx = archive["fine_centroid_x"].astype(np.float32)
                cy = archive["fine_centroid_y"].astype(np.float32)
                area = archive["fine_area"].astype(np.float32)
            region_labels = np.load(labels.loc[row["patch_id"], "label_path"]).astype(np.int64)
            valid = active & (region_labels != args.ignore_index)
            adjacency = centroid_knn_adjacency(cx, cy, active, args.centroid_knn)
            present = sorted(set(map(int, region_labels[valid])) & set(args.class_ids))
            rng = np.random.default_rng(args.seed + len(rows))
            patch_pixels = max(int(np.rint(area[active].sum())), 1)
            for size_name, spec in PROMPT_SIZE_SPECS.items():
                for target in present:
                    if int((valid & (region_labels != target)).sum()) < int(spec["negative_slots"]):
                        continue
                    try:
                        slots = sample_connected_region_set(
                            adjacency,
                            valid & (region_labels == target),
                            np.maximum(np.rint(area), 1).astype(np.int64),
                            rng,
                            min_slots=int(spec["min_slots"]),
                            max_slots=int(spec["max_slots"]),
                            min_fraction=float(spec["min_fraction"]),
                            max_fraction=float(spec["max_fraction"]),
                            patch_pixels=patch_pixels,
                        )
                    except ValueError:
                        continue
                    episode_row = {
                            "patch_id": row["patch_id"],
                            "wsi_id": row["wsi_id"],
                            "split": split,
                            "sampling_group": row["sampling_group"],
                            "prompt_size": size_name,
                            "target_class": target,
                            "positive_slots": slots.tolist(),
                            "positive_fraction": float(area[slots].sum() / patch_pixels),
                        }
                    rows.append(episode_row)
                    patch_episode_rows.append(episode_row)
            with episode_records.open("a", encoding="utf-8") as handle:
                for episode_row in patch_episode_rows:
                    handle.write(json.dumps(episode_row) + "\n")
            with completed.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"split": split, "patch_id": row["patch_id"]}) + "\n")
        counts[split] = int(len(part))

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("no feasible prompt episodes found")
    table.to_parquet(output / "eligibility_index.parquet", index=False)
    summary = {
        "timestamp": stamp,
        "cache_index": str(args.cache_index),
        "label_index": str(args.label_index),
        "test_used": "test" in args.splits,
        "scanned_patches": counts,
        "eligible_episode_rows": len(table),
        "by_split": table["split"].value_counts().to_dict(),
        "by_size": table["prompt_size"].value_counts().to_dict(),
        "by_class": {str(k): int(v) for k, v in table["target_class"].value_counts().sort_index().items()},
        "size_specs": PROMPT_SIZE_SPECS,
        "centroid_knn": args.centroid_knn,
        "seed": args.seed,
    }
    write_json(output / "metadata.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
