#!/usr/bin/env python3
"""Audit a Phase-5 eligibility manifest against its source cache and labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.phase_5_prompt_encoder.src.prompts import (
    PROMPT_SIZE_SPECS,
    _is_connected,
    centroid_knn_adjacency,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eligibility-index", type=Path, required=True)
    p.add_argument("--completed", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--label-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples-per-stratum", type=int, default=3)
    p.add_argument("--fraction-tolerance", type=float, default=1e-6)
    return p.parse_args()


def main():
    a = parse_args()
    ep = pd.read_parquet(a.eligibility_index)
    cache = pd.read_parquet(a.cache_index)
    labels = pd.read_parquet(a.label_index).set_index("patch_id")
    completed = pd.DataFrame(json.loads(line) for line in a.completed.read_text().splitlines())
    joined = ep.merge(cache[["patch_id", "split"]], on="patch_id", suffixes=("_episode", "_cache"))
    report = {
        "rows": len(ep),
        "null_counts": {k: int(v) for k, v in ep.isna().sum().items()},
        "duplicate_episode_keys": int(ep.duplicated(["patch_id", "prompt_size", "target_class"]).sum()),
        "completed_rows": len(completed),
        "completed_duplicate_patch_ids": int(completed.duplicated("patch_id").sum()),
        "completed_split_counts": completed["split"].value_counts().to_dict(),
        "unique_eligible_patches": int(ep["patch_id"].nunique()),
        "patches_without_episode": int(cache["patch_id"].nunique() - ep["patch_id"].nunique()),
        "split_mismatch_with_cache": int((joined["split_episode"] != joined["split_cache"]).sum()),
        "wsi_split_overlap": int((ep.groupby("wsi_id")["split"].nunique() > 1).sum()),
    }
    report["size_class_matrix"] = {}
    for split in ("train", "val"):
        matrix = pd.crosstab(ep.loc[ep["split"] == split, "target_class"], ep.loc[ep["split"] == split, "prompt_size"])
        matrix = matrix.reindex(index=range(12), columns=list(PROMPT_SIZE_SPECS), fill_value=0)
        report["size_class_matrix"][split] = {str(k): {n: int(v) for n, v in row.items()} for k, row in matrix.to_dict("index").items()}
    for size, spec in PROMPT_SIZE_SPECS.items():
        part = ep[ep["prompt_size"] == size]
        lengths = part["positive_slots"].map(len)
        bad = ((lengths < spec["min_slots"]) | (lengths > spec["max_slots"]) | (part["positive_fraction"] < spec["min_fraction"] - a.fraction_tolerance) | (part["positive_fraction"] > spec["max_fraction"] + a.fraction_tolerance))
        report[size] = {"slot_range": [int(lengths.min()), int(lengths.max())], "fraction_range": [float(part["positive_fraction"].min()), float(part["positive_fraction"].max())], "fraction_tolerance": a.fraction_tolerance, "bound_violations": int(bad.sum())}

    sampled_indices = []
    for _, stratum in ep.groupby(["split", "prompt_size", "target_class"], sort=True):
        sampled_indices.extend(
            stratum.sample(min(a.samples_per_stratum, len(stratum)), random_state=20260720).index.tolist()
        )
    sample = ep.loc[sampled_indices].reset_index(drop=True)
    cache = cache.set_index("patch_id")
    checks = {"rows": len(sample), "label_mismatch": 0, "inactive_slot": 0, "disconnected": 0, "fraction_mismatch": 0, "insufficient_negatives": 0}
    started = time.monotonic()
    for row in sample.to_dict("records"):
        source = cache.loc[row["patch_id"]]
        with np.load(source["shard_path"]) as z:
            active = z["fine_active"].astype(bool); cx = z["fine_centroid_x"]; cy = z["fine_centroid_y"]; area = z["fine_area"].astype(float)
        lab = np.load(labels.loc[row["patch_id"], "label_path"]).astype(int)
        slots = np.asarray(row["positive_slots"], dtype=int)
        checks["label_mismatch"] += int(not np.all(lab[slots] == int(row["target_class"])))
        checks["inactive_slot"] += int(not np.all(active[slots]))
        checks["disconnected"] += int(not _is_connected(slots, centroid_knn_adjacency(cx, cy, active, 4)))
        fraction = float(area[slots].sum() / max(area[active].sum(), 1e-6))
        checks["fraction_mismatch"] += int(not np.isclose(fraction, row["positive_fraction"], rtol=1e-5, atol=1e-7))
        required = PROMPT_SIZE_SPECS[row["prompt_size"]]["negative_slots"]
        checks["insufficient_negatives"] += int(np.sum(active & (lab != 255) & (lab != int(row["target_class"]))) < required)
    checks["read_seconds"] = time.monotonic() - started
    report["source_replay"] = checks
    report["passed"] = not any([
        sum(report["null_counts"].values()), report["duplicate_episode_keys"], report["completed_duplicate_patch_ids"],
        report["split_mismatch_with_cache"], report["wsi_split_overlap"],
        *[report[size]["bound_violations"] for size in PROMPT_SIZE_SPECS],
        checks["label_mismatch"], checks["inactive_slot"], checks["disconnected"], checks["fraction_mismatch"], checks["insufficient_negatives"],
    ]) and report["completed_rows"] == len(cache)
    a.output.parent.mkdir(parents=True, exist_ok=False)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
