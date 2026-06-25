from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import openslide

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.fullres_inference import FullResolutionEngine, TOKEN_ORDER
from benchmarks.gdph_v2.geometry import generate_core_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one full-resolution pilot tile.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--xcell_checkpoint",
        default="experiments/crc012_holdout3/20260515_160711/he_model_best.pth",
    )
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--cellpose_checkpoint", default="/home/zyh/.cellpose/models/cpsam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile_size", type=int, default=4096)
    parser.add_argument("--halo", type=int, default=256)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file))
    slide = openslide.OpenSlide(row["he_path"])
    width, height = slide.level_dimensions[0]
    tile = generate_core_tiles(width, height, args.tile_size, args.halo)[0]
    tile_rgb = np.asarray(
        slide.read_region(
            (tile.read_x0, tile.read_y0),
            0,
            (tile.read_x1 - tile.read_x0, tile.read_y1 - tile.read_y0),
        ).convert("RGB")
    )
    slide.close()

    started = time.time()
    engine = FullResolutionEngine(
        args.xcell_checkpoint,
        args.ctranspath_checkpoint,
        args.cellpose_checkpoint,
        args.device,
    )
    cells = engine.process_tile(tile_rgb)
    keep = np.asarray(
        [
            tile.owns(x + tile.read_x0, y + tile.read_y0)
            for x, y in cells.centers_xy
        ]
    )
    report = {
        "image_id": row["image_id"],
        "tile_shape": list(tile_rgb.shape),
        "detected_cells": len(cells.labels),
        "owned_cells": int(keep.sum()),
        "raw_shape": list(cells.raw.shape),
        "reg_shape": list(cells.reg.shape),
        "proj_shape": list(cells.proj.shape),
        "polygons": len(cells.polygons),
        "xcell_token_order": TOKEN_ORDER,
        "labels_strictly_increasing": bool(
            len(cells.labels) < 2 or np.all(np.diff(cells.labels) > 0)
        ),
        "finite": all(np.isfinite(array).all() for array in (cells.raw, cells.reg, cells.proj)),
        "elapsed_seconds": time.time() - started,
    }
    report["passed"] = (
        report["detected_cells"] > 0
        and report["detected_cells"]
        == report["raw_shape"][0]
        == report["reg_shape"][0]
        == report["proj_shape"][0]
        == report["polygons"]
        and report["finite"]
        and report["labels_strictly_increasing"]
    )
    output_path = Path(args.output_root) / "config" / "fullres_smoke_validation.json"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("full-resolution smoke test failed")


if __name__ == "__main__":
    main()
