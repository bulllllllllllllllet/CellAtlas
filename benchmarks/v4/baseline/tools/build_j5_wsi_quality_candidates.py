#!/usr/bin/env python3
"""Build Dice-blind, quality-controlled J5 WSI prompt candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--eligible-quantile", type=float, default=0.75)
    parser.add_argument("--min-target-fraction", type=float, default=0.001)
    parser.add_argument("--purity-weight", type=float, default=0.25)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def global_point(row: pd.Series, column: str) -> dict[str, float]:
    local = json.loads(row[column])
    downsample = float(row["level0_downsample"])
    return {
        "x": float(row["x_level0"]) + float(local[0]) * downsample,
        "y": float(row["y_level0"]) + float(local[1]) * downsample,
    }


def choose_spatially_diverse(
    pool: pd.DataFrame,
    count: int,
    purity_weight: float,
) -> list[int]:
    if len(pool) < count:
        raise RuntimeError(f"quality-controlled pool has {len(pool)} rows, requires {count}")
    xy = np.stack(
        [
            (pool["x_level0"] + pool["width_level0"] / 2) / pool["wsi_width_level0"],
            (pool["y_level0"] + pool["height_level0"] / 2) / pool["wsi_height_level0"],
        ],
        axis=1,
    )
    purity = pool["target_fraction"].to_numpy(float)
    purity = (purity - purity.min()) / max(float(np.ptp(purity)), 1e-12)
    selected = [int(np.argmax(purity))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(xy[:, None, :] - xy[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        score = distances + purity_weight * purity
        score[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def main() -> None:
    args = parse_args()
    if args.candidates < 2:
        raise ValueError("--candidates must be at least 2")
    if not 0.0 <= args.eligible_quantile < 1.0:
        raise ValueError("--eligible-quantile must be in [0, 1)")
    if not 0.0 <= args.min_target_fraction <= 1.0:
        raise ValueError("--min-target-fraction must be in [0, 1]")
    if args.purity_weight < 0.0:
        raise ValueError("--purity-weight must be non-negative")

    output = args.output_root / f"j5_wsi_quality_candidates_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    task_frames = [pd.read_parquet(path) for path in args.task_manifest]
    tasks = pd.concat(task_frames, ignore_index=True)
    identity = ["wsi_id", "target_class_name"]
    if tasks.duplicated(identity).any():
        duplicate = tasks.loc[tasks.duplicated(identity, keep=False), identity]
        raise ValueError(f"duplicate WSI-class tasks:\n{duplicate.to_string(index=False)}")

    rows: list[dict[str, object]] = []
    task_audits: list[dict[str, object]] = []
    for task in tasks.sort_values(["target_class", "wsi_id"]).to_dict("records"):
        patch_manifest = Path(task["patch_prompt_manifest"])
        frame = pd.read_parquet(patch_manifest).copy()
        required = {
            "source_index",
            "patch_id",
            "has_target",
            "negative_available",
            "positive_audit",
            "negative_audit",
            "target_pixels",
            "valid_pixels",
            "positive_point_10x",
            "negative_point_10x",
            "x_level0",
            "y_level0",
            "width_level0",
            "height_level0",
            "wsi_width_level0",
            "wsi_height_level0",
            "level0_downsample",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{patch_manifest} is missing columns {sorted(missing)}")
        eligible = frame.loc[
            frame["has_target"]
            & frame["negative_available"]
            & frame["positive_audit"]
            & frame["negative_audit"]
        ].copy()
        eligible["target_fraction"] = eligible["target_pixels"] / eligible["valid_pixels"]
        quantile_threshold = float(eligible["target_fraction"].quantile(args.eligible_quantile))
        threshold = max(args.min_target_fraction, quantile_threshold)
        pool = eligible.loc[eligible["target_fraction"].ge(threshold)].copy()
        pool = pool.sort_values("source_index").reset_index(drop=True)
        minimum_pool_size = math.ceil((1.0 - args.eligible_quantile) * len(eligible))
        if len(pool) < args.candidates:
            raise RuntimeError(
                f"{task['wsi_id']} {task['target_class_name']}: only {len(pool)} candidates "
                f"pass target_fraction >= {threshold:.8f}; requires {args.candidates}"
            )
        selected = choose_spatially_diverse(pool, args.candidates, args.purity_weight)
        task_audits.append(
            {
                "wsi_id": str(task["wsi_id"]),
                "target_class": int(task["target_class"]),
                "target_class_name": str(task["target_class_name"]),
                "eligible_patch_count": int(len(eligible)),
                "quality_pool_count": int(len(pool)),
                "quantile_threshold": quantile_threshold,
                "effective_threshold": threshold,
                "theoretical_quantile_pool_minimum": minimum_pool_size,
                "patch_prompt_manifest": str(patch_manifest),
                "patch_prompt_manifest_sha256": sha256_path(patch_manifest),
            }
        )
        for candidate_index, pool_index in enumerate(selected):
            row = pool.iloc[pool_index]
            prompt = {
                "coordinate_space": "level0",
                "prompt_size": "point",
                "positive": [global_point(row, "positive_point_10x")],
                "negative": [global_point(row, "negative_point_10x")],
            }
            safe_class = str(task["target_class_name"]).replace("/", "_")
            prompt_path = output / (
                f"{safe_class}__{task['wsi_id']}__candidate_{candidate_index:02d}"
                f"__{args.timestamp}.json"
            )
            prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
            rows.append(
                {
                    "wsi_id": str(task["wsi_id"]),
                    "target_class": int(task["target_class"]),
                    "target_class_name": str(task["target_class_name"]),
                    "candidate_index": candidate_index,
                    "selection_rule": (
                        "dice_blind_top_quartile_target_fraction_then_spatial_diversity"
                    ),
                    "source_index": int(row["source_index"]),
                    "patch_id": str(row["patch_id"]),
                    "target_pixels": int(row["target_pixels"]),
                    "valid_pixels": int(row["valid_pixels"]),
                    "target_fraction": float(row["target_fraction"]),
                    "quality_threshold": threshold,
                    "positive_audit": bool(row["positive_audit"]),
                    "negative_audit": bool(row["negative_audit"]),
                    "prompt_json": str(prompt_path),
                    "prompt_json_sha256": sha256_path(prompt_path),
                    "tile_index": str(task["tile_index"]),
                    "gt_path": str(task["gt_path"]),
                    "wsi_path": str(task["wsi_path"]),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["target_class", "wsi_id", "candidate_index"]
    ).reset_index(drop=True)
    expected_rows = len(tasks) * args.candidates
    if len(result) != expected_rows:
        raise RuntimeError(f"candidate count mismatch: {len(result)} != {expected_rows}")
    if result.duplicated(["wsi_id", "target_class_name", "candidate_index"]).any():
        raise RuntimeError("duplicate WSI-class candidate identity")
    if not result["positive_audit"].all() or not result["negative_audit"].all():
        raise RuntimeError("candidate semantic audit failed")
    if not result["target_fraction"].ge(result["quality_threshold"]).all():
        raise RuntimeError("candidate target-fraction audit failed")

    combined_manifest = output / f"candidate_manifest_all_{args.timestamp}.parquet"
    result.to_parquet(combined_manifest, index=False)
    class_manifests: dict[str, str] = {}
    for class_name, class_frame in result.groupby("target_class_name", sort=True):
        class_manifest = output / f"candidate_manifest_{class_name}_{args.timestamp}.parquet"
        class_frame.to_parquet(class_manifest, index=False)
        class_manifests[str(class_name)] = str(class_manifest)

    audit_frame = pd.DataFrame(task_audits).sort_values(
        ["target_class", "wsi_id"]
    ).reset_index(drop=True)
    audit_manifest = output / f"quality_audit_{args.timestamp}.parquet"
    audit_frame.to_parquet(audit_manifest, index=False)
    metadata = {
        "timestamp": args.timestamp,
        "protocol": "quality-controlled five-prompt WSI evaluation",
        "dice_blind": True,
        "quality_rule": {
            "semantic_checks": "positive inside target GT; negative outside target GT",
            "eligible_quantile": args.eligible_quantile,
            "minimum_target_fraction": args.min_target_fraction,
            "effective_threshold": "max(per-WSI-class quantile, minimum_target_fraction)",
            "selection": "greedy spatial diversity with target-fraction preference",
            "purity_weight": args.purity_weight,
        },
        "task_manifests": [str(path) for path in args.task_manifest],
        "task_count": int(len(tasks)),
        "wsi_count": int(tasks["wsi_id"].nunique()),
        "class_count": int(tasks["target_class_name"].nunique()),
        "candidate_count_per_task": args.candidates,
        "candidate_count_total": int(len(result)),
        "combined_manifest": str(combined_manifest),
        "combined_manifest_sha256": sha256_path(combined_manifest),
        "class_manifests": class_manifests,
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": sha256_path(audit_manifest),
        "disclosure": (
            "GT-guided quality-controlled prompts; report as simulated clean-click "
            "performance, not random-click robustness."
        ),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "combined_manifest": str(combined_manifest),
                "audit_manifest": str(audit_manifest),
                "rows": len(result),
                "task_count": len(tasks),
                "minimum_selected_target_fraction": float(result["target_fraction"].min()),
                "minimum_quality_pool_size": int(audit_frame["quality_pool_count"].min()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
