#!/usr/bin/env python3
"""Create a one-tile, coordinate-consistent J5 inference smoke from the TLS case."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyvips

from module.KFBreader.kfbreader import KFBSlide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", type=Path, required=True)
    parser.add_argument("--cell-feature-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=530)
    parser.add_argument("--positive-x", type=float, default=46772.0)
    parser.add_argument("--positive-y", type=float, default=16394.0)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/test_TLS"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def find_feature_record(manifest: Path, source_index: int) -> tuple[dict, pa.Schema]:
    index = pd.read_parquet(manifest)
    match = index[(index["start"] <= source_index) & (index["end"] > source_index)]
    if len(match) != 1:
        raise RuntimeError(f"expected one feature shard for source {source_index}, found {len(match)}")
    table = pq.read_table(str(match.iloc[0]["shard_path"]))
    records = [row for row in table.to_pylist() if int(row["source_index"]) == source_index]
    if len(records) != 1:
        raise RuntimeError(f"expected one feature record for source {source_index}, found {len(records)}")
    return records[0], table.schema


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"tls_inference_smoke_input_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    rows = pd.read_parquet(args.tile_index)
    selected = rows[rows["source_index"] == args.source_index]
    if len(selected) != 1:
        raise RuntimeError(f"expected one tile row for source {args.source_index}, found {len(selected)}")
    source = selected.iloc[0].to_dict()
    x0 = int(source["x_level0"]); y0 = int(source["y_level0"])
    width = int(source["width_level0"]); height = int(source["height_level0"])
    local_positive = (args.positive_x - x0, args.positive_y - y0)
    if not (0 <= local_positive[0] < width and 0 <= local_positive[1] < height):
        raise ValueError("positive prompt does not lie inside the selected smoke tile")

    slide = KFBSlide(str(source["wsi_path"]))
    image = np.asarray(slide.read_region((x0, y0), 0, (width, height)), dtype=np.uint8)[..., :3]
    image_path = output / "he_level0.tif"
    pyvips.Image.new_from_memory(
        memoryview(image), width, height, 3, "uchar"
    ).tiffsave(str(image_path), tile=True, tile_width=512, tile_height=512, compression="deflate")

    patch_id = f"TLS_SMOKE_{args.source_index}"
    smoke_row = {
        "source_index": 0, "patch_id": patch_id, "wsi_id": "TLS_SMOKE", "split": "val",
        "x_10x": 0, "y_10x": 0, "width_10x": int(source["width_10x"]),
        "height_10x": int(source["height_10x"]), "x_level0": 0, "y_level0": 0,
        "width_level0": width, "height_level0": height,
        "level0_downsample": int(source["level0_downsample"]),
        "wsi_width_level0": width, "wsi_height_level0": height,
        "output_width_10x": int(source["width_10x"]),
        "output_height_10x": int(source["height_10x"]), "wsi_path": str(image_path),
    }
    tile_index_path = output / "wsi_tile_index.parquet"
    pd.DataFrame([smoke_row]).to_parquet(tile_index_path, index=False)

    feature, schema = find_feature_record(args.cell_feature_manifest, args.source_index)
    feature["source_index"] = 0; feature["patch_id"] = patch_id; feature["wsi_id"] = "TLS_SMOKE"
    shard_path = output / "features_0000000_0000000.parquet"
    pq.write_table(pa.Table.from_pylist([feature], schema=schema), shard_path, compression="zstd")
    feature_index_path = output / "feature_index.parquet"
    pd.DataFrame([{
        "shard_path": str(shard_path), "start": 0, "end": 1, "rows": 1,
    }]).to_parquet(feature_index_path, index=False)

    prompt_path = output / "prompts.json"
    prompts = {
        "coordinate_space": "level0", "prompt_size": "large",
        "positive": [{"x": float(local_positive[0]), "y": float(local_positive[1])}],
        "negative": [{"x": float(width - 128), "y": float(height - 128)}],
    }
    prompt_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    metadata = {
        "timestamp": stamp, "source_index": args.source_index,
        "source_level0_origin": [x0, y0], "source_tile_path": str(source["wsi_path"]),
        "image_path": str(image_path), "tile_index": str(tile_index_path),
        "cell_feature_manifest": str(feature_index_path), "prompt_json": str(prompt_path),
        "purpose": "mechanical one-tile J5 smoke only; not a TLS retrieval result",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
