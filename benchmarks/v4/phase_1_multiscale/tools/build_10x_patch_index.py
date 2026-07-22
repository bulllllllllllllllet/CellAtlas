#!/usr/bin/env python3
"""Build a dynamic 10x patch index without exporting image patches.

Each WSI is persisted as an independent Parquet shard immediately after it is
processed.  A final index is merged sequentially from those shards only after
every WSI succeeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json
from benchmarks.v4.phase_1_multiscale.src.data import boundary_mask, decode_gt


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Deterministic manifest-row sample for validation.")
    parser.add_argument("--wsi-id", help="Process exactly one named WSI for a validation run.")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--timestamp")
    parser.add_argument("--resume-output", type=Path, help="Continue an interrupted timestamped patch-index artifact directory.")
    return parser.parse_args()


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def sampling_group(present: set[int], boundary_fraction: float, tissue_fraction: float, rare: set[int], special: set[int], threshold: float) -> str:
    if present & rare:
        return "rare_class"
    if boundary_fraction >= threshold:
        return "class_boundary"
    if present & special:
        return "low_cell_or_special_tissue"
    if tissue_fraction < 0.5:
        return "background_or_hard_negative"
    return "class_interior"


def build_slide_shard(slide: dict, config: dict, rare: set[int], shard_dir: str) -> dict:
    shard_dir = Path(shard_dir)
    data, sampling = config["data"], config["sampling"]
    ignore = int(data["ignore_index"]); size, stride = int(data["patch_size"]), int(data["index_stride"])
    background = set(map(int, data["background_class_ids"])); classes = [int(item["id"]) for item in data["class_map"]]
    special = set(map(int, sampling["special_class_ids"])); threshold = float(sampling["boundary_fraction_threshold"])
    gt = decode_gt(Path(slide["gt_path"]), data["class_map"], ignore)
    edge = boundary_mask(gt, ignore)
    sx, sy = float(slide["inferred_gt_downsample_x"]), float(slide["inferred_gt_downsample_y"])
    rows: list[dict] = []
    for y in range(0, gt.shape[0] - size + 1, stride):
        for x in range(0, gt.shape[1] - size + 1, stride):
            patch = gt[y:y + size, x:x + size]
            valid = patch != ignore
            valid_fraction = float(valid.mean())
            tissue_fraction = float((valid & ~np.isin(patch, list(background))).mean())
            if valid_fraction < float(data["min_valid_fraction"]) or tissue_fraction < float(data["min_tissue_fraction"]):
                continue
            counts = {str(cls): int((patch == cls).sum()) for cls in classes}
            present = {cls for cls in classes if counts[str(cls)]}
            dominant = max(present, key=lambda cls: counts[str(cls)])
            boundary_fraction = float(edge[y:y + size, x:x + size].mean())
            rows.append({
                "patch_id": f"{slide['wsi_id']}__x{x}_y{y}", "wsi_id": slide["wsi_id"], "patient_id": slide["patient_id"], "split": slide["split"],
                "x_10x": x, "y_10x": y, "width_10x": size, "height_10x": size,
                "x_level0": round(x * sx), "y_level0": round(y * sy), "width_level0": round(size * sx), "height_level0": round(size * sy),
                "target_mpp": None, "valid_fraction": valid_fraction, "tissue_fraction": tissue_fraction, "boundary_fraction": boundary_fraction,
                "present_classes": json.dumps(sorted(present)), "class_pixel_counts": json.dumps(counts), "dominant_class": dominant,
                "sampling_group": sampling_group(present, boundary_fraction, tissue_fraction, rare, special, threshold),
                "wsi_path": slide["wsi_path"], "gt_path": slide["gt_path"],
            })
    shard = shard_dir / f"patches_{slide['wsi_id']}.parquet"
    temporary = shard_dir / f"{shard.stem}.attempt_{uuid4().hex}.parquet.inprogress"
    if shard.exists():
        raise FileExistsError(f"refusing to overwrite index shard: {shard}")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.rename(shard)
    return {"wsi_id": slide["wsi_id"], "shard": shard.name, "patch_count": len(rows)}


def merge_shards(shards: list[Path], output: Path) -> int:
    writer = None; count = 0
    try:
        for shard in sorted(shards):
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table); count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no non-empty patch shards to merge")
    return count


def main() -> None:
    args = parse(); config = load_config(args.config)
    if args.num_workers < 1: raise ValueError("--num-workers must be positive")
    if args.resume_output:
        output = args.resume_output.resolve()
        if not output.is_dir(): raise FileNotFoundError(f"resume artifact directory does not exist: {output}")
    else:
        output = create_run_dir(config, "patch_index", args.timestamp)
    slides = pd.read_parquet(args.manifest).to_dict("records")
    if args.wsi_id:
        slides = [slide for slide in slides if slide["wsi_id"] == args.wsi_id]
        if len(slides) != 1: raise ValueError(f"WSI not found exactly once: {args.wsi_id}")
    if args.limit:
        if not 1 <= args.limit <= len(slides): raise ValueError(f"--limit must be in 1..{len(slides)}")
        rng = np.random.default_rng(args.seed)
        slides = [slides[i] for i in sorted(rng.choice(len(slides), size=args.limit, replace=False))]
    train_pixels = {int(item["id"]): 0 for item in config["data"]["class_map"]}
    for slide in slides:
        if slide["split"] != "train": continue
        for cls, value in slide["class_pixel_counts"].items(): train_pixels[int(cls)] += int(value)
    total = max(sum(train_pixels.values()), 1)
    rare = {cls for cls, value in train_pixels.items() if value / total <= float(config["sampling"]["rare_class_max_train_fraction"])}
    shard_dir = output / "shards"; shard_dir.mkdir(exist_ok=args.resume_output is not None)
    completed_path, failures_path = output / "completed.jsonl", output / "failures.jsonl"
    completed_ids: set[str] = set(); shards: list[Path] = []
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line); shard = shard_dir / record["shard"]
            if not shard.is_file(): raise RuntimeError(f"completed record has no shard: {shard}")
            completed_ids.add(record["wsi_id"]); shards.append(shard)
    # A stop can occur between atomic shard publication and JSONL append. Reuse
    # those valid shards instead of recomputing the WSI, while recording it.
    for slide in slides:
        shard = shard_dir / f"patches_{slide['wsi_id']}.parquet"
        if slide["wsi_id"] not in completed_ids and shard.is_file():
            count = pq.ParquetFile(shard).metadata.num_rows
            record = {"wsi_id": slide["wsi_id"], "shard": shard.name, "patch_count": count, "status": "reconciled"}
            append_jsonl(completed_path, record); completed_ids.add(slide["wsi_id"]); shards.append(shard)
    slides = [slide for slide in slides if slide["wsi_id"] not in completed_ids]
    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(build_slide_shard, slide, config, rare, str(shard_dir)): slide["wsi_id"] for slide in slides}
        for future in as_completed(futures):
            wsi_id = futures[future]
            try:
                result = future.result(); shards.append(shard_dir / result["shard"]); append_jsonl(completed_path, result)
            except Exception as exc:
                failure = {"wsi_id": wsi_id, "error": str(exc)}; failures.append(failure); append_jsonl(failures_path, failure)
    if failures:
        save_json(output / "patch_index_failures.json", failures)
        raise RuntimeError(f"{len(failures)} WSI failed; successful shards remain in {shard_dir}")
    index_path = output / "patch_index_10x.parquet"
    patch_count = merge_shards(shards, index_path)
    save_json(output / "patch_index_metadata.json", {
        "input_manifest": str(args.manifest), "slide_count": len(completed_ids) + len(slides), "patch_count": patch_count,
        "num_workers": args.num_workers, "rare_classes": sorted(rare), "shards": [path.name for path in sorted(shards)], "seed": args.seed,
        "resumed": args.resume_output is not None,
    })
    print(index_path)


if __name__ == "__main__":
    main()
