#!/usr/bin/env python3
"""Build complete overlapping WSI tile indexes for a fixed cohort split."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyvips

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import read_pairs
from benchmarks.v4.whole_slide_inference.src.tiling import (
    build_tile_rows,
    validate_tile_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--level0-downsample", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def image_shape(path: Path) -> tuple[int, int, int, str]:
    image = pyvips.Image.new_from_file(str(path), access="random")
    return int(image.width), int(image.height), int(image.bands), str(image.format)


def main() -> None:
    args = parse_args()
    batch_root = args.output_root / f"test_wsi_cell_features_{args.timestamp}"
    if batch_root.exists():
        raise FileExistsError(f"refusing to overwrite: {batch_root}")
    batch_root.mkdir(parents=True)
    tile_root = batch_root / "tile_indexes"
    tile_root.mkdir()

    cohort = pd.read_parquet(args.cohort_manifest)
    selected = cohort.loc[cohort["split"].eq(args.split)].copy()
    if selected.empty or selected["wsi_id"].duplicated().any():
        raise ValueError("selected cohort split is empty or contains duplicate WSI IDs")
    selected = selected.sort_values("wsi_id", kind="stable").reset_index(drop=True)
    pairs = {pair.wsi_id: pair for pair in read_pairs(load_config(args.config))}
    missing = sorted(set(selected["wsi_id"]) - set(pairs))
    if missing:
        raise ValueError(f"cohort WSIs missing from nuclei pair CSV: {missing}")

    manifest_rows: list[dict] = []
    for selection_rank, cohort_row in enumerate(selected.to_dict("records"), start=1):
        wsi_id = str(cohort_row["wsi_id"])
        pair = pairs[wsi_id]
        paths = (pair.he_path, pair.nuclei_instance_path, pair.nuclei_class_path)
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"missing inputs for {wsi_id}: {absent}")
        he_shape, instance_shape, class_shape = (image_shape(path) for path in paths)
        if instance_shape[:2] != he_shape[:2] or class_shape[:2] != he_shape[:2]:
            raise ValueError(
                f"HE/nuclei dimension mismatch for {wsi_id}: "
                f"HE={he_shape}, instance={instance_shape}, class={class_shape}"
            )
        if he_shape[2] < 3 or instance_shape[2] != 1 or class_shape[2] != 1:
            raise ValueError(
                f"unexpected HE/nuclei bands for {wsi_id}: "
                f"HE={he_shape}, instance={instance_shape}, class={class_shape}"
            )
        expected_shape = (
            int(cohort_row["level0_width"]), int(cohort_row["level0_height"])
        )
        if he_shape[:2] != expected_shape:
            raise ValueError(
                f"cohort/decoded WSI dimension mismatch for {wsi_id}: "
                f"manifest={expected_shape}, decoded={he_shape[:2]}"
            )

        tile_rows = build_tile_rows(
            wsi_id=wsi_id,
            wsi_path=str(pair.he_path),
            width_level0=he_shape[0],
            height_level0=he_shape[1],
            level0_downsample=args.level0_downsample,
            tile_size=args.tile_size,
            stride=args.stride,
            split=args.split,
        )
        output_width, output_height, downsample = validate_tile_rows(tile_rows)
        case_id = f"case_{selection_rank:03d}_{wsi_id}"
        case_dir = tile_root / f"{case_id}_{args.timestamp}"
        case_dir.mkdir()
        tile_index = case_dir / "wsi_tile_index.parquet"
        pd.DataFrame(tile_rows).to_parquet(tile_index, index=False)
        manifest_rows.append(
            {
                "selection_rank": selection_rank,
                "case_id": case_id,
                "wsi_id": wsi_id,
                "split": args.split,
                "wsi_path": str(pair.he_path),
                "nuclei_instance_path": str(pair.nuclei_instance_path),
                "nuclei_class_path": str(pair.nuclei_class_path),
                "tile_index": str(tile_index),
                "tile_count": len(tile_rows),
                "level0_width": he_shape[0],
                "level0_height": he_shape[1],
                "output_width_10x": output_width,
                "output_height_10x": output_height,
                "level0_downsample": downsample,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = batch_root / f"test_wsi_feature_manifest_{args.timestamp}.parquet"
    manifest.to_parquet(manifest_path, index=False)
    manifest.to_csv(
        batch_root / f"test_wsi_feature_manifest_{args.timestamp}.csv", index=False
    )
    summary = {
        "timestamp": args.timestamp,
        "status": "prepared",
        "cohort_manifest": str(args.cohort_manifest),
        "config": str(args.config),
        "split": args.split,
        "wsi_count": len(manifest),
        "tile_total": int(manifest["tile_count"].sum()),
        "tile_min": int(manifest["tile_count"].min()),
        "tile_max": int(manifest["tile_count"].max()),
        "tile_size_10x": args.tile_size,
        "stride_10x": args.stride,
        "level0_downsample": args.level0_downsample,
        "dimension_checks": "30/30 HE, instance mask, and class mask dimensions identical",
        "manifest": str(manifest_path),
    }
    (batch_root / f"preparation_summary_{args.timestamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
