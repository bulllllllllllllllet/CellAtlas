#!/usr/bin/env python3
"""Render deterministic HE/GT/overlay samples from the dynamic patch index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt, read_he_patch


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def colorize(mask: np.ndarray, class_map: list[dict], ignore: int) -> np.ndarray:
    output = np.full((*mask.shape, 3), 128, dtype=np.uint8)
    for item in class_map:
        output[mask == int(item["id"])] = np.asarray(item["rgb"], dtype=np.uint8)
    output[mask == ignore] = (96, 96, 96)
    return output


def main() -> None:
    args = parse()
    config = load_config(args.config)
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    rows = pd.read_parquet(args.index)
    if len(rows) < args.num_samples:
        raise ValueError("index contains fewer rows than requested samples")
    # One sample from each deterministically selected WSI avoids inspecting two
    # adjacent tiles from the same source image.
    rng = np.random.default_rng(args.seed)
    chosen_wsis = rng.choice(rows["wsi_id"].unique(), size=args.num_samples, replace=False)
    selected = []
    for wsi_id in chosen_wsis:
        candidates = rows.loc[rows["wsi_id"] == wsi_id]
        selected.append(candidates.iloc[int(rng.integers(len(candidates)))].to_dict())
    output = create_run_dir(config, "patch_index_verify", args.timestamp)
    ignore = int(config["data"]["ignore_index"])
    results = []
    for sample_id, row in enumerate(selected):
        image = read_he_patch(Path(row["wsi_path"]), int(row["x_level0"]), int(row["y_level0"]), int(row["width_level0"]), int(row["height_level0"]), int(row["width_10x"]))
        whole_mask = decode_gt(Path(row["gt_path"]), config["data"]["class_map"], ignore)
        mask = whole_mask[int(row["y_10x"]):int(row["y_10x"] + row["height_10x"]), int(row["x_10x"]):int(row["x_10x"] + row["width_10x"])]
        if image.shape[:2] != mask.shape:
            raise RuntimeError(f"HE/GT shape mismatch for {row['patch_id']}: {image.shape} vs {mask.shape}")
        allowed = {int(item["id"]) for item in config["data"]["class_map"]} | {ignore}
        observed = set(np.unique(mask).tolist())
        if not observed <= allowed:
            raise RuntimeError(f"invalid class values for {row['patch_id']}: {sorted(observed - allowed)}")
        labels = colorize(mask, config["data"]["class_map"], ignore)
        overlay = (0.55 * image.astype(np.float32) + 0.45 * labels.astype(np.float32)).round().astype(np.uint8)
        panel = np.concatenate([image, labels, overlay], axis=1)
        filename = f"sample_{sample_id:02d}_{row['wsi_id']}_x{row['x_10x']}_y{row['y_10x']}.png"
        Image.fromarray(panel).save(output / filename)
        results.append({"file": filename, "patch_id": row["patch_id"], "wsi_id": row["wsi_id"], "split": row["split"], "shape": list(image.shape), "observed_class_ids": sorted(map(int, observed)), "checks": {"he_gt_shape_equal": True, "class_values_valid": True}})
    save_json(output / "verification.json", {"index": str(args.index), "seed": args.seed, "samples": results, "panel_layout": "HE | GT palette | 45% GT overlay"})
    print(output)


if __name__ == "__main__":
    main()
