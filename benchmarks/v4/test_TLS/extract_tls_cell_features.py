#!/usr/bin/env python3
"""Build v4-compatible per-tile Cellpose/CTransPath/XCell features for a KFB WSI."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from benchmarks.v4.phase_3_cell_region.tools.extract_xcell_features import (
    FEATURE_SCHEMA,
    extract_raw_reference_compatible,
    load_models,
    selected_cells_from_maps,
)
from module.KFBreader.kfbreader import KFBSlide
from new_inference_stream.inference import is_patch_valid
from utils import load_cellpose_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/test_TLS"))
    parser.add_argument("--ctranspath-checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--xcell-checkpoint", default="module/checkpoint/he_model_best.pth")
    parser.add_argument("--cellpose-model", default="/home/zhaoyh/.cellpose/models/cpsam")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--xcell-batch-size", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=255)
    parser.add_argument("--cell-batch-size", type=int, default=255)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def empty_record(row: dict) -> dict:
    return {
        "patch_id": str(row["patch_id"]), "wsi_id": str(row["wsi_id"]),
        "total_cell_count": 0, "source_index": int(row["source_index"]),
        "cells": [], "reg_features": [], "proj_features": [],
    }


def extract_one_pre_xcell(
    row: dict, slide: KFBSlide, ctp, cellpose, args, device
) -> tuple[dict, np.ndarray, dict]:
    started = time.perf_counter()
    x, y, width, height = (int(row[key]) for key in ("x_level0", "y_level0", "width_level0", "height_level0"))
    he = np.asarray(slide.read_region((x, y), 0, (width, height)), dtype=np.uint8)[..., :3]
    read_sec = time.perf_counter() - started
    if not is_patch_valid(he):
        return empty_record(row), np.empty((0, 768), np.float32), {
            "read_sec": read_sec, "cellpose_sec": 0.0, "cell_count": 0,
        }
    started = time.perf_counter()
    masks = cellpose.eval(he, diameter=18, channels=[0, 0])[0]
    cellpose_sec = time.perf_counter() - started
    classes = np.zeros_like(masks, dtype=np.uint8)
    metadata, selected_ids, total = selected_cells_from_maps(
        np.asarray(masks, np.int32), classes, args.max_cells, "spatial_stratified", 8,
    )
    if not len(selected_ids):
        return empty_record(row), np.empty((0, 768), np.float32), {
            "read_sec": read_sec, "cellpose_sec": cellpose_sec, "cell_count": 0,
        }
    raw, profile = extract_raw_reference_compatible(
        he, np.asarray(masks, np.int32), selected_ids, ctp, device,
        args.cell_batch_size, "batched",
    )
    if len(raw) != len(metadata):
        raise RuntimeError(f"cell metadata/feature mismatch {len(metadata)} != {len(raw)}")
    profile.update({
        "read_sec": read_sec, "cellpose_sec": cellpose_sec,
        "cell_count": int(total),
    })
    record = {
        "patch_id": str(row["patch_id"]), "wsi_id": str(row["wsi_id"]),
        "total_cell_count": int(total), "source_index": int(row["source_index"]),
        "cells": metadata.tolist(), "reg_features": [], "proj_features": [],
    }
    return record, raw, profile


def run_xcell_batches(records: list[dict], raw_features: list[np.ndarray], xcell, args, device) -> float:
    """Stack independent 255-cell tensors along B, never across the cell axis."""
    elapsed = 0.0
    nonempty = [index for index, raw in enumerate(raw_features) if len(raw)]
    for start in range(0, len(nonempty), args.xcell_batch_size):
        batch_indices = nonempty[start:start + args.xcell_batch_size]
        part_raw = [raw_features[index] for index in batch_indices]
        padded = np.zeros((len(part_raw), args.max_cells, 768), np.float32)
        valid = np.zeros((len(part_raw), args.max_cells), np.float32)
        for index, raw in enumerate(part_raw):
            padded[index, :len(raw)] = raw
            valid[index, :len(raw)] = 1
        started = time.perf_counter()
        with torch.inference_mode():
            _, reg, proj, _ = xcell(
                raw_images=None, x=torch.from_numpy(padded).to(device),
                mask=torch.from_numpy(valid).to(device),
            )
        elapsed += time.perf_counter() - started
        reg = reg.float().cpu().numpy()
        proj = proj.float().cpu().numpy()
        for local, raw in enumerate(part_raw):
            record_index = batch_indices[local]
            records[record_index]["reg_features"] = reg[local, :len(raw)].tolist()
            records[record_index]["proj_features"] = proj[local, :len(raw)].tolist()
    return elapsed


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"tls_cell_features_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    rows = pd.read_parquet(args.tile_index).sort_values("source_index").to_dict("records")
    end = len(rows) if args.end is None else int(args.end)
    if args.start < 0 or end <= args.start or end > len(rows):
        raise ValueError(f"invalid range {args.start}:{end} for {len(rows)} tiles")
    selected = rows[args.start:end]
    if not selected:
        raise ValueError("no selected tiles")
    if args.xcell_batch_size < 1:
        raise ValueError("xcell_batch_size must be positive")
    if len({row["wsi_path"] for row in selected}) != 1:
        raise ValueError("TLS extraction expects exactly one WSI")
    output.mkdir(parents=True)
    completed_path = output / "completed.jsonl"
    failures_path = output / "failures.jsonl"
    device = torch.device(args.device)
    ctp, xcell = load_models(args, device)
    cellpose = load_cellpose_model(args.cellpose_model, device)
    slide = KFBSlide(str(selected[0]["wsi_path"]))
    refs, timings = [], []
    with completed_path.open("a", encoding="utf-8") as completed, failures_path.open("a", encoding="utf-8") as failures:
        for local_start in range(0, len(selected), args.shard_size):
            part = selected[local_start:local_start + args.shard_size]
            payload = []
            raw_features = []
            part_profiles = []
            try:
                for row in part:
                    value, raw, profile = extract_one_pre_xcell(
                        row, slide, ctp, cellpose, args, device,
                    )
                    payload.append(value)
                    raw_features.append(raw)
                    part_profiles.append(profile)
                xcell_elapsed = run_xcell_batches(payload, raw_features, xcell, args, device)
                xcell_per_tile = xcell_elapsed / max(len(payload), 1)
                for row, profile in zip(part, part_profiles, strict=True):
                    timings.append({
                        "source_index": int(row["source_index"]), "patch_id": row["patch_id"],
                        **profile, "xcell_sec_amortized": xcell_per_tile,
                    })
            except Exception as exc:
                failures.write(json.dumps({"source_index": int(row["source_index"]), "error": repr(exc)}) + "\n")
                failures.flush()
                raise
            first = int(part[0]["source_index"]); stop = int(part[-1]["source_index"]) + 1
            shard = output / f"features_{first:07d}_{stop - 1:07d}.parquet"
            pq.write_table(pa.Table.from_pylist(payload, schema=FEATURE_SCHEMA), shard, compression="zstd")
            reference = {"shard_path": str(shard), "start": first, "end": stop, "rows": len(part)}
            refs.append(reference)
            completed.write(json.dumps(reference) + "\n"); completed.flush()
            print({"event": "shard_complete", "done": local_start + len(part), "total": len(selected)}, flush=True)
    pd.DataFrame(refs).to_parquet(output / "feature_index.parquet", index=False)
    pd.DataFrame(timings).to_parquet(output / "timing_by_patch.parquet", index=False)
    metadata = {
        "timestamp": stamp, "tile_index": str(args.tile_index), "start": args.start, "end": end,
        "rows": len(selected), "complete_tile_index": args.start == 0 and end == len(rows),
        "cellpose_model": args.cellpose_model, "ctranspath_checkpoint": args.ctranspath_checkpoint,
        "xcell_checkpoint": args.xcell_checkpoint, "max_cells": args.max_cells,
        "xcell_batch_size": args.xcell_batch_size,
        "selection_policy": "spatial_stratified", "nuclei_class": "unknown=0 for HE-only Cellpose",
        "feature_contract": "[x_norm,y_norm,log1p_area,class_id] + XCellFormer reg-64",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
