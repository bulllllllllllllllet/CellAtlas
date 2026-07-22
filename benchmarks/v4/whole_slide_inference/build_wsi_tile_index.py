#!/usr/bin/env python3
"""Build a complete overlapping 10x tile index for one WSI."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyvips

from benchmarks.v4.whole_slide_inference.src.tiling import build_tile_rows, validate_tile_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--wsi-id", required=True)
    parser.add_argument("--level0-downsample", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/nfs-medical3/zyh/v4/whole_slide_inference/tile_indexes"),
    )
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wsi_path.is_file():
        raise FileNotFoundError(args.wsi_path)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"wsi_tile_index_{args.wsi_id}_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    slide = pyvips.Image.new_from_file(str(args.wsi_path), access="random")
    rows = build_tile_rows(
        args.wsi_id, str(args.wsi_path), slide.width, slide.height,
        args.level0_downsample, args.tile_size, args.stride, args.split,
    )
    width, height, downsample = validate_tile_rows(rows)
    output.mkdir(parents=True)
    index_path = output / "wsi_tile_index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    metadata = {
        "timestamp": stamp,
        "wsi_id": args.wsi_id,
        "wsi_path": str(args.wsi_path),
        "wsi_width_level0": int(slide.width),
        "wsi_height_level0": int(slide.height),
        "output_width_10x": width,
        "output_height_10x": height,
        "level0_downsample": downsample,
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "tile_count": len(rows),
        "split": args.split,
        "tile_index": str(index_path),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
