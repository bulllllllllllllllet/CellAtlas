#!/usr/bin/env python3
"""Bind a fixed cohort's dynamic patch index to lossless tiled GT files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--tiled-gt-manifest", type=Path, required=True)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse(); config = load_config(args.config)
    cohort = pd.read_parquet(args.cohort_manifest)
    tiled = pd.read_parquet(args.tiled_gt_manifest)[["wsi_id", "tiled_gt_path"]]
    if cohort["wsi_id"].duplicated().any() or tiled["wsi_id"].duplicated().any():
        raise ValueError("cohort or tiled GT manifest has duplicate WSI ids")
    mapping = cohort[["wsi_id", "split"]].merge(tiled, on="wsi_id", how="left", validate="one_to_one")
    if mapping["tiled_gt_path"].isna().any():
        raise RuntimeError(f"missing tiled GT for {int(mapping['tiled_gt_path'].isna().sum())} cohort WSI")
    missing_files = [path for path in mapping["tiled_gt_path"] if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"tiled GT files missing, first: {missing_files[0]}")
    index = pd.read_parquet(args.patch_index)
    output_index = index.merge(mapping, on=["wsi_id", "split"], how="inner", validate="many_to_one")
    if output_index["wsi_id"].nunique() != len(cohort):
        raise RuntimeError("cohort WSI missing from patch index")
    output_index["gt_path"] = output_index.pop("tiled_gt_path")
    output = create_run_dir(config, "tiled_cohort_patch_index", args.timestamp)
    path = output / "patch_index_10x_tiled_gt.parquet"
    output_index.to_parquet(path, index=False)
    save_json(output / "metadata.json", {"source_patch_index": str(args.patch_index), "cohort_manifest": str(args.cohort_manifest), "tiled_gt_manifest": str(args.tiled_gt_manifest), "patch_index": str(path), "wsi_count": int(output_index["wsi_id"].nunique()), "patch_count": len(output_index), "split_counts": output_index.groupby("split").size().to_dict()})
    print(path)


if __name__ == "__main__":
    main()
