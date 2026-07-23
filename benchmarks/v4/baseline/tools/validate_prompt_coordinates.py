#!/usr/bin/env python3
"""Audit local/level-0 coordinate transforms and render representative prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch
from benchmarks.v4.baseline.common import atomic_json, new_output_directory, parse_json_array, sha256_path, timestamp, validate_episode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--occurrence-id")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    frame = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
    audit = validate_episode_manifest(frame, args.split)
    # Coordinate equality is checked algebraically for every occurrence.
    sx = frame["width_level0"].to_numpy(float) / frame["width_10x"].to_numpy(float)
    sy = frame["height_level0"].to_numpy(float) / frame["height_10x"].to_numpy(float)
    if not np.isfinite(sx).all() or not np.isfinite(sy).all() or (sx <= 0).any() or (sy <= 0).any():
        raise ValueError("invalid level-0/10x scale factors")
    if float(np.max(np.abs(sx - sy))) > 1e-3:
        raise ValueError("anisotropic level-0/10x mapping exceeds tolerance")
    selected = frame.iloc[0] if args.occurrence_id is None else frame.loc[frame["occurrence_id"] == args.occurrence_id].iloc[0]
    cfg = load_config(args.phase2_config); ignore = int(cfg["data"]["ignore_index"])
    image = read_he_patch(
        Path(selected.wsi_path), int(selected.x_level0), int(selected.y_level0),
        int(selected.width_level0), int(selected.height_level0), int(selected.width_10x),
    )
    gt = decode_gt_patch(
        Path(selected.gt_path), int(selected.x_10x), int(selected.y_10x),
        int(selected.width_10x), int(selected.height_10x), cfg["data"]["class_map"], ignore,
    )
    if image.shape[:2] != gt.shape:
        raise RuntimeError(f"HE/GT shape mismatch {image.shape} vs {gt.shape}")
    positive = parse_json_array(selected.positive_points_10x, 2)
    negative = parse_json_array(selected.negative_points_10x, 2)
    truth = gt == int(selected.target_class)
    output = new_output_directory(args.output_root, "coordinate_audit", stamp)
    matplotlib_cache = output / f"matplotlib_cache_{stamp}"
    matplotlib_cache.mkdir()
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panel = output / f"coordinate_panel_{stamp}.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image); axes[0].set_title("HE + frozen prompts")
    axes[1].imshow(truth, cmap="gray"); axes[1].set_title("GT (audit only)")
    axes[2].imshow(image); axes[2].imshow(truth, alpha=0.35, cmap="spring"); axes[2].set_title("HE / GT overlay")
    for ax in axes:
        if len(positive): ax.scatter(positive[:, 0], positive[:, 1], s=60, facecolors="none", edgecolors="lime", linewidths=2)
        if len(negative): ax.scatter(negative[:, 0], negative[:, 1], s=55, marker="x", color="red", linewidths=2)
        if selected.prompt_size != "point":
            x0, y0, x1, y1 = json.loads(selected.positive_box_10x)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
        ax.axis("off")
    fig.savefig(panel, dpi=140); plt.close(fig)
    report = {
        "timestamp": stamp, "manifest": str(args.episode_manifest), "manifest_sha256": sha256_path(args.episode_manifest),
        "manifest_audit": audit, "scale_x_range": [float(sx.min()), float(sx.max())],
        "scale_y_range": [float(sy.min()), float(sy.max())], "max_scale_axis_difference": float(np.max(np.abs(sx - sy))),
        "representative_occurrence": str(selected.occurrence_id), "image_shape": list(image.shape),
        "gt_shape": list(gt.shape), "panel": str(panel), "status": "passed",
    }
    atomic_json(output / f"coordinate_audit_{stamp}.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
