#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyvips
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json, stable_hash
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt, read_pairs


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a deterministic, explicit 10x WSI/GT manifest.")
    p.add_argument("--config", type=Path, required=True); p.add_argument("--timestamp")
    p.add_argument("--limit", type=int, help="Deterministic sample size; omit for every explicit pair.")
    p.add_argument("--exclude-limit", type=int, help="Exclude this deterministic sample size before processing.")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260714)
    return p.parse_args()


def build_row(pair, class_map: list[dict], ignore: int, unknown_policy: str) -> dict:
    slide = pyvips.Image.new_from_file(str(pair.he_path), access="random")
    gt = decode_gt(pair.gt_path, class_map, ignore)
    sx, sy = slide.width / gt.shape[1], slide.height / gt.shape[0]
    status = "aligned" if abs(sx - sy) / max(sx, sy) < 0.01 else "anisotropic_or_misaligned"
    if status != "aligned": raise ValueError(f"x/y scale mismatch: {sx:.6f} vs {sy:.6f}")
    valid = gt != ignore
    unknown_pixels = int((~valid).sum())
    if unknown_pixels and unknown_policy != "ignore":
        raise ValueError(f"unknown GT colors: {unknown_pixels} pixels")
    counts = {str(c["id"]): int((gt == int(c["id"])).sum()) for c in class_map}
    dominant = int(max(counts, key=counts.get))
    return {"wsi_id": pair.wsi_id, "patient_id": pair.patient_id, "wsi_path": str(pair.he_path), "gt_path": str(pair.gt_path), "center": None, "scanner": None, "level0_width": slide.width, "level0_height": slide.height, "base_mpp_x": None, "base_mpp_y": None, "objective_power": None, "available_levels": 1, "level_downsamples": "[1.0]", "gt_width": int(gt.shape[1]), "gt_height": int(gt.shape[0]), "gt_dtype": str(gt.dtype), "gt_unique_values": sorted(map(int, np.unique(gt))), "inferred_gt_downsample_x": sx, "inferred_gt_downsample_y": sy, "alignment_status": status, "class_pixel_counts": counts, "dominant_class": dominant, "valid_pixels": int(valid.sum()), "unknown_pixel_count": unknown_pixels}


def split_rows(rows: list[dict], cfg: dict) -> list[dict]:
    ratios = cfg["split"]; targets = {k: ratios[k] * len(rows) for k in ("train", "val", "test")}
    counts = {k: 0 for k in targets}
    # deterministic dominant-class round robin minimizes split-size error without patch-level leakage
    for dominant in sorted({r["dominant_class"] for r in rows}):
        for row in sorted((r for r in rows if r["dominant_class"] == dominant), key=lambda r: r["wsi_id"]):
            split = min(targets, key=lambda k: (counts[k] / max(targets[k], 1e-9), k))
            row["split"] = split; counts[split] += 1
    return rows


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_shard(shard_dir: Path, shard_id: int, rows: list[dict]) -> Path:
    path = shard_dir / f"manifest_rows_{shard_id:05d}.parquet"
    if path.exists(): raise FileExistsError(f"refusing to overwrite shard: {path}")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def main() -> None:
    ns = args(); cfg = load_config(ns.config)
    out = create_run_dir(cfg, "manifest", ns.timestamp)
    class_map, ignore = cfg["data"]["class_map"], int(cfg["data"]["ignore_index"])
    unknown_policy = cfg["data"].get("unknown_color_policy", "error")
    if unknown_policy not in {"error", "ignore"}: raise ValueError("data.unknown_color_policy must be error or ignore")
    if not class_map: raise ValueError("manifest split requires exact data.class_map after audit")
    pairs = read_pairs(cfg)
    if ns.num_workers < 1 or ns.shard_size < 1: raise ValueError("--num-workers and --shard-size must be positive")
    if ns.exclude_limit:
        if not 1 <= ns.exclude_limit < len(pairs): raise ValueError(f"--exclude-limit must be in 1..{len(pairs) - 1}")
        rng = np.random.default_rng(ns.seed)
        excluded = set(rng.choice(len(pairs), size=ns.exclude_limit, replace=False).tolist())
        pairs = [pair for i, pair in enumerate(pairs) if i not in excluded]
    if ns.limit:
        if not 1 <= ns.limit <= len(pairs): raise ValueError(f"--limit must be in 1..{len(pairs)}")
        rng = np.random.default_rng(ns.seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=ns.limit, replace=False))]
    shard_dir = out / "shards"; shard_dir.mkdir()
    completed_path, failures_path = out / "completed.jsonl", out / "failures.jsonl"
    pending, failures, shards, shard_id = [], [], [], 0
    with ProcessPoolExecutor(max_workers=ns.num_workers) as pool:
        futures = {pool.submit(build_row, pair, class_map, ignore, unknown_policy): pair.wsi_id for pair in pairs}
        for future in as_completed(futures):
            wsi_id = futures[future]
            try:
                row = future.result(); pending.append(row)
                append_jsonl(completed_path, {"wsi_id": wsi_id, "status": "ok"})
                if len(pending) >= ns.shard_size:
                    shards.append(write_shard(shard_dir, shard_id, pending)); shard_id += 1; pending = []
            except Exception as exc:
                failure = {"wsi_id": wsi_id, "error": str(exc)}; failures.append(failure); append_jsonl(failures_path, failure)
    if pending: shards.append(write_shard(shard_dir, shard_id, pending))
    if failures:
        save_json(out / "manifest_failures.json", failures)
        raise RuntimeError(f"{len(failures)} invalid slides; successful rows are retained in {shard_dir}")
    rows = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True).to_dict("records")
    rows = split_rows(rows, cfg)
    frame = pd.DataFrame(rows); manifest, split = out / "wsi_gt_manifest.parquet", out / "patient_split.csv"
    frame.to_parquet(manifest, index=False); frame[["wsi_id", "patient_id", "split"]].to_csv(split, index=False)
    save_json(out / "manifest_metadata.json", {"timestamp": out.name.rsplit("_", 2)[-2] + "_" + out.name.rsplit("_", 1)[-1], "manifest": str(manifest), "split_hash": stable_hash(frame[["wsi_id", "patient_id", "split"]].to_dict("records")), "class_map": class_map, "sample_limit": ns.limit, "exclude_limit": ns.exclude_limit, "num_workers": ns.num_workers, "shard_size": ns.shard_size, "shards": [path.name for path in shards], "seed": ns.seed})
    print(manifest)


if __name__ == "__main__": main()
