#!/usr/bin/env python3
"""Build deterministic spatially diverse GT-guided J5 best-of-K WSI prompts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def global_point(row: pd.Series, column: str) -> dict[str, float]:
    local = json.loads(row[column])
    downsample = float(row["level0_downsample"])
    return {
        "x": float(row["x_level0"]) + float(local[0]) * downsample,
        "y": float(row["y_level0"]) + float(local[1]) * downsample,
    }


def choose_diverse(frame: pd.DataFrame, count: int, excluded: set[int]) -> list[int]:
    eligible = frame.loc[
        frame["has_target"] & frame["negative_available"] & ~frame["source_index"].isin(excluded)
    ].copy()
    eligible["target_fraction"] = eligible["target_pixels"] / eligible["valid_pixels"]
    threshold = float(eligible["target_fraction"].median())
    pool = eligible.loc[eligible["target_fraction"] >= threshold].copy()
    if len(pool) < count:
        raise RuntimeError(f"only {len(pool)} eligible high-content patches for {count} candidates")
    xy = np.stack([
        (pool["x_level0"] + pool["width_level0"] / 2) / pool["wsi_width_level0"],
        (pool["y_level0"] + pool["height_level0"] / 2) / pool["wsi_height_level0"],
    ], axis=1)
    purity = pool["target_fraction"].to_numpy(float)
    purity = (purity - purity.min()) / max(float(np.ptp(purity)), 1e-12)
    selected: list[int] = []
    first = int(np.argmax(purity))
    selected.append(first)
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(xy[:, None, :] - xy[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        score = distances + 0.25 * purity
        score[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(score)))
    return pool.iloc[selected]["source_index"].astype(int).tolist()


def main() -> None:
    args = parse_args()
    if args.candidates < 2:
        raise ValueError("--candidates must be at least 2")
    tasks = pd.read_parquet(args.task_manifest)
    output = args.output_root / f"j5_wsi_oracle_candidates_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for task in tasks.to_dict("records"):
        frame = pd.read_parquet(task["patch_prompt_manifest"]).set_index("source_index", drop=False)
        original = json.loads(Path(task["j5_prompt_json"]).read_text(encoding="utf-8"))
        original_source = int(frame.loc[frame["patch_id"].eq(task["seed_patch_id"]), "source_index"].iloc[0])
        selections = [original_source] + choose_diverse(
            frame.reset_index(drop=True), args.candidates - 1, {original_source}
        )
        for candidate_index, source_index in enumerate(selections):
            row = frame.loc[source_index]
            if candidate_index == 0:
                prompt = original
                rule = "frozen_single_random"
            else:
                prompt = {
                    "coordinate_space": "level0",
                    "prompt_size": "point",
                    "positive": [global_point(row, "positive_point_10x")],
                    "negative": [global_point(row, "negative_point_10x")],
                }
                rule = "greedy_spatial_diversity_high_target_fraction"
            prompt_path = output / f"{task['wsi_id']}_candidate_{candidate_index:02d}_{args.timestamp}.json"
            prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
            rows.append({
                "wsi_id": str(task["wsi_id"]),
                "target_class": int(task["target_class"]),
                "target_class_name": str(task["target_class_name"]),
                "candidate_index": candidate_index,
                "selection_rule": rule,
                "source_index": source_index,
                "patch_id": str(row["patch_id"]),
                "target_pixels": int(row["target_pixels"]),
                "valid_pixels": int(row["valid_pixels"]),
                "target_fraction": float(row["target_pixels"] / row["valid_pixels"]),
                "positive_audit": bool(row["positive_audit"]),
                "negative_audit": bool(row["negative_audit"]),
                "prompt_json": str(prompt_path),
                "tile_index": str(task["tile_index"]),
                "gt_path": str(task["gt_path"]),
                "wsi_path": str(task["wsi_path"]),
            })
    manifest = output / f"candidate_manifest_{args.timestamp}.parquet"
    result = pd.DataFrame(rows)
    if len(result) != len(tasks) * args.candidates:
        raise RuntimeError("candidate count mismatch")
    if not result["positive_audit"].all() or not result["negative_audit"].all():
        raise RuntimeError("candidate semantic audit failed")
    result.to_parquet(manifest, index=False)
    (output / "metadata.json").write_text(json.dumps({
        "timestamp": args.timestamp,
        "protocol": f"best-of-{args.candidates} GT-oracle WSI seed selection",
        "candidate_manifest": str(manifest),
        "wsi_count": len(tasks),
        "candidate_count_per_wsi": args.candidates,
        "total_full_wsi_inferences": len(result),
        "disclosure": "Oracle result; do not report as single-interaction performance.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest), "rows": len(result)}, indent=2))


if __name__ == "__main__":
    main()
