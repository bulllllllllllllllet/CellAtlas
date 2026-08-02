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
from cellpose import models as cellpose_models
from cellpose import transforms as cellpose_transforms
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
    parser.add_argument(
        "--tile-batch-size", type=int, default=1,
        help="Number of same-sized WSI tiles passed to one Cellpose eval call.",
    )
    parser.add_argument(
        "--cellpose-net-batch-size", type=int, default=8,
        help="Cellpose internal network batch size for 256x256 model blocks.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Validate and continue an existing contiguous shard prefix in the timestamped output.",
    )
    parser.add_argument("--timestamp")
    return parser.parse_args()


def empty_record(row: dict) -> dict:
    return {
        "patch_id": str(row["patch_id"]), "wsi_id": str(row["wsi_id"]),
        "total_cell_count": 0, "source_index": int(row["source_index"]),
        "cells": [], "reg_features": [], "proj_features": [],
    }


def read_tile(row: dict, slide: KFBSlide) -> tuple[np.ndarray, float, bool]:
    started = time.perf_counter()
    x, y, width, height = (int(row[key]) for key in ("x_level0", "y_level0", "width_level0", "height_level0"))
    he = np.asarray(slide.read_region((x, y), 0, (width, height)), dtype=np.uint8)[..., :3]
    read_sec = time.perf_counter() - started
    return he, read_sec, bool(is_patch_valid(he))


def extract_one_from_mask(
    row: dict, he: np.ndarray, masks: np.ndarray, ctp, args, device,
    read_sec: float, cellpose_sec: float,
) -> tuple[dict, np.ndarray, dict]:
    if masks.size == 0:
        return empty_record(row), np.empty((0, 768), np.float32), {
            "read_sec": read_sec, "cellpose_sec": 0.0, "cell_count": 0,
        }
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


def extract_tile_batch_pre_xcell(
    rows: list[dict], slide: KFBSlide, ctp, cellpose, args, device,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    images, read_seconds, valid = [], [], []
    for row in rows:
        he, read_sec, is_valid = read_tile(row, slide)
        images.append(he)
        read_seconds.append(read_sec)
        valid.append(is_valid)
    valid_indices = [index for index, value in enumerate(valid) if value]
    masks_by_index: dict[int, np.ndarray] = {}
    cellpose_per_tile = 0.0
    if valid_indices:
        started = time.perf_counter()
        if len(valid_indices) == 1:
            masks = cellpose.eval(
                images[valid_indices[0]], diameter=18, channels=[0, 0],
                batch_size=args.cellpose_net_batch_size,
            )[0]
            masks_by_index[valid_indices[0]] = np.asarray(masks, np.int32)
        else:
            masks = cellpose_eval_batch_2d(
                cellpose,
                np.stack([images[index] for index in valid_indices]),
                diameter=18.0,
                net_batch_size=args.cellpose_net_batch_size,
            )
            if len(masks) != len(valid_indices):
                raise RuntimeError(
                    f"batched Cellpose cardinality mismatch {len(masks)} != {len(valid_indices)}"
                )
            for index, mask in zip(valid_indices, masks, strict=True):
                masks_by_index[index] = np.asarray(mask, np.int32)
        cellpose_per_tile = (time.perf_counter() - started) / len(valid_indices)
    records, raw_features, profiles = [], [], []
    for index, (row, he, read_sec) in enumerate(zip(rows, images, read_seconds, strict=True)):
        mask = masks_by_index.get(index, np.empty((0, 0), np.int32))
        record, raw, profile = extract_one_from_mask(
            row, he, mask, ctp, args, device, read_sec, cellpose_per_tile,
        )
        records.append(record)
        raw_features.append(raw)
        profiles.append(profile)
    return records, raw_features, profiles


def cellpose_eval_batch_2d(
    cellpose, images: np.ndarray, diameter: float, net_batch_size: int,
) -> np.ndarray:
    """Batch the public Cellpose-4 2D eval math without its list-level serial loop."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected [B,H,W,3] images, got {images.shape}")
    if len(images) < 2:
        raise ValueError("batched Cellpose path requires at least two images")
    height, width = map(int, images.shape[1:3])
    image_scaling = 30.0 / diameter
    resized = cellpose_transforms.resize_image(
        images,
        Ly=int(height * image_scaling),
        Lx=int(width * image_scaling),
    )
    normalize_params = dict(cellpose_models.normalize_default)
    normalize_params["norm3D"] = False
    normalized = cellpose_transforms.normalize_img(resized, **normalize_params)
    dP, cellprob, _ = cellpose._run_net(
        normalized,
        augment=False,
        batch_size=net_batch_size,
        tile_overlap=0.1,
        bsize=256,
        do_3D=False,
        anisotropy=None,
    )
    dP = np.stack([
        cellpose._resize_gradients(
            dP[:, index], to_y_size=height, to_x_size=width,
        )
        for index in range(len(images))
    ], axis=1)
    cellprob = np.stack([
        cellpose._resize_cellprob(
            cellprob[index], to_x_size=width, to_y_size=height,
        )
        for index in range(len(images))
    ])
    masks = []
    for index in range(len(images)):
        mask = cellpose._compute_masks(
            normalized[index:index + 1].shape,
            dP[:, index:index + 1],
            cellprob[index:index + 1],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=15,
            max_size_fraction=0.4,
            niter=int(200 / image_scaling),
            stitch_threshold=0.0,
            do_3D=False,
        )
        masks.append(np.asarray(mask).squeeze())
    masks = np.stack(masks)
    masks = cellpose_transforms.resize_image(
        masks, Ly=height, Lx=width, no_channels=True, interpolation=0,
    )
    if masks.shape != images.shape[:3]:
        raise RuntimeError(
            f"batched Cellpose output shape {masks.shape} != {images.shape[:3]}"
        )
    return masks.astype(np.int32, copy=False)


def existing_prefix(output: Path, selected: list[dict]) -> tuple[list[dict], int]:
    completed_path = output / "completed.jsonl"
    if not completed_path.is_file():
        raise FileNotFoundError(f"resume requested but missing {completed_path}")
    refs = [
        json.loads(line) for line in completed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = int(selected[0]["source_index"])
    covered = 0
    for ref in refs:
        if int(ref["start"]) != expected:
            raise ValueError(f"non-contiguous resume prefix at {ref}")
        shard = Path(ref["shard_path"])
        if not shard.is_file():
            raise FileNotFoundError(shard)
        table = pq.read_table(shard, columns=["source_index"])
        if table.num_rows != int(ref["rows"]) or int(ref["end"]) - int(ref["start"]) != table.num_rows:
            raise ValueError(f"invalid resume shard {shard}")
        expected = int(ref["end"])
        covered += table.num_rows
    if covered > len(selected):
        raise ValueError(f"resume prefix {covered} exceeds selected rows {len(selected)}")
    expected_indices = [int(row["source_index"]) for row in selected[:covered]]
    observed_indices = []
    for ref in refs:
        observed_indices.extend(
            pq.read_table(ref["shard_path"], columns=["source_index"])
            .column("source_index").to_pylist()
        )
    if observed_indices != expected_indices:
        raise ValueError("resume shard source_index sequence differs from tile index")
    return refs, covered


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
    rows = pd.read_parquet(args.tile_index).sort_values("source_index").to_dict("records")
    end = len(rows) if args.end is None else int(args.end)
    if args.start < 0 or end <= args.start or end > len(rows):
        raise ValueError(f"invalid range {args.start}:{end} for {len(rows)} tiles")
    selected = rows[args.start:end]
    if not selected:
        raise ValueError("no selected tiles")
    if args.xcell_batch_size < 1:
        raise ValueError("xcell_batch_size must be positive")
    if args.tile_batch_size < 1 or args.cellpose_net_batch_size < 1:
        raise ValueError("tile-batch-size and cellpose-net-batch-size must be positive")
    if len({row["wsi_path"] for row in selected}) != 1:
        raise ValueError("TLS extraction expects exactly one WSI")
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=args.resume)
    completed_path = output / "completed.jsonl"
    failures_path = output / "failures.jsonl"
    device = torch.device(args.device)
    ctp, xcell = load_models(args, device)
    cellpose = load_cellpose_model(args.cellpose_model, device)
    slide = KFBSlide(str(selected[0]["wsi_path"]))
    refs, covered = existing_prefix(output, selected) if args.resume else ([], 0)
    timings = []
    with completed_path.open("a", encoding="utf-8") as completed, failures_path.open("a", encoding="utf-8") as failures:
        for local_start in range(covered, len(selected), args.shard_size):
            part = selected[local_start:local_start + args.shard_size]
            payload = []
            raw_features = []
            part_profiles = []
            try:
                for batch_start in range(0, len(part), args.tile_batch_size):
                    batch_rows = part[batch_start:batch_start + args.tile_batch_size]
                    batch_payload, batch_raw, batch_profiles = extract_tile_batch_pre_xcell(
                        batch_rows, slide, ctp, cellpose, args, device,
                    )
                    payload.extend(batch_payload)
                    raw_features.extend(batch_raw)
                    part_profiles.extend(batch_profiles)
                xcell_elapsed = run_xcell_batches(payload, raw_features, xcell, args, device)
                xcell_per_tile = xcell_elapsed / max(len(payload), 1)
                for row, profile in zip(part, part_profiles, strict=True):
                    timings.append({
                        "source_index": int(row["source_index"]), "patch_id": row["patch_id"],
                        **profile, "xcell_sec_amortized": xcell_per_tile,
                    })
            except Exception as exc:
                failures.write(json.dumps({
                    "source_index": int(part[0]["source_index"]), "error": repr(exc),
                }) + "\n")
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
        "tile_batch_size": args.tile_batch_size,
        "cellpose_net_batch_size": args.cellpose_net_batch_size,
        "resumed_rows": covered,
        "selection_policy": "spatial_stratified", "nuclei_class": "unknown=0 for HE-only Cellpose",
        "feature_contract": "[x_norm,y_norm,log1p_area,class_id] + XCellFormer reg-64",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
