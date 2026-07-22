#!/usr/bin/env python3
"""Convert immutable GT PNG labels to lossless, 512-pixel tiled BigTIFF files."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyvips

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wsi-id", help="Convert one named WSI for validation.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rgb_crop(image: pyvips.Image, x: int, y: int, width: int, height: int) -> np.ndarray:
    crop = image.crop(x, y, width, height).extract_band(0, n=3)
    return np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(height, width, 3))


def verify_lossless(source: Path, converted: Path) -> dict:
    original = pyvips.Image.new_from_file(str(source), access="random")
    tiled = pyvips.Image.new_from_file(str(converted), access="random")
    if (original.width, original.height, original.bands) != (tiled.width, tiled.height, tiled.bands):
        raise RuntimeError("converted TIFF dimensions or bands differ from source PNG")
    positions = [(0, 0), (max(0, (original.width - 512) // 2), max(0, (original.height - 512) // 2)), (max(0, original.width - 512), max(0, original.height - 512))]
    for x, y in positions:
        width, height = min(512, original.width - x), min(512, original.height - y)
        if not np.array_equal(rgb_crop(original, x, y, width, height), rgb_crop(tiled, x, y, width, height)):
            raise RuntimeError(f"RGB mismatch at {(x, y, width, height)}")
    return {"width": original.width, "height": original.height, "bands": original.bands, "verified_crops": len(positions)}


def convert_one(row: dict, output_dir: str) -> dict:
    started = time.monotonic(); root = Path(output_dir); source = Path(row["gt_path"])
    target = root / "tiles" / f"{row['wsi_id']}.tif"
    temporary = target.with_suffix(".tif.inprogress")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite tiled GT: {target}")
    image = pyvips.Image.new_from_file(str(source), access="sequential")
    image.tiffsave(str(temporary), tile=True, tile_width=512, tile_height=512, compression="lzw", bigtiff=True, pyramid=False)
    temporary.rename(target)
    verification = verify_lossless(source, target)
    return {"wsi_id": row["wsi_id"], "source_gt_path": str(source), "tiled_gt_path": str(target), "elapsed_seconds": time.monotonic() - started, "file_bytes": target.stat().st_size, "verification": verification}


def main() -> None:
    args = parse(); config = load_config(args.config)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.resume_output:
        output = args.resume_output.resolve()
        if not output.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {output}")
    else:
        output = create_run_dir(config, "tiled_gt", args.timestamp)
        (output / "tiles").mkdir()
    rows = pd.read_parquet(args.manifest).to_dict("records")
    if args.wsi_id:
        rows = [row for row in rows if row["wsi_id"] == args.wsi_id]
        if len(rows) != 1:
            raise ValueError(f"WSI not found exactly once: {args.wsi_id}")
    completed_path, failures_path = output / "completed.jsonl", output / "failures.jsonl"
    completed_ids: set[str] = set()
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not Path(record["tiled_gt_path"]).is_file():
                raise RuntimeError(f"completed record has no tiled file: {record['tiled_gt_path']}")
            completed_ids.add(record["wsi_id"])
    pending = [row for row in rows if row["wsi_id"] not in completed_ids]
    failures: list[dict] = []; records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(convert_one, row, str(output)): row["wsi_id"] for row in pending}
        for future in as_completed(futures):
            wsi_id = futures[future]
            try:
                record = future.result(); records.append(record); append_jsonl(completed_path, record)
            except Exception as exc:
                failure = {"wsi_id": wsi_id, "error": str(exc)}; failures.append(failure); append_jsonl(failures_path, failure)
    if failures:
        save_json(output / "conversion_failures.json", failures)
        raise RuntimeError(f"{len(failures)} GT conversions failed; successful tiled files remain")
    completed = [json.loads(line) for line in completed_path.read_text(encoding="utf-8").splitlines()]
    pd.DataFrame(completed).to_parquet(output / "tiled_gt_manifest.parquet", index=False)
    save_json(output / "tiled_gt_metadata.json", {"source_manifest": str(args.manifest), "slide_count": len(completed), "num_workers": args.num_workers, "tile_size": 512, "compression": "lzw", "lossless_validation": "three RGB crop comparisons per WSI", "resumed": bool(args.resume_output)})
    print(output / "tiled_gt_manifest.parquet")


if __name__ == "__main__":
    main()
