#!/usr/bin/env python3
"""Build fine-scale region GT labels for cached Phase-4 patches (10x Phase2 + mask)."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch  # libvips first
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor, load_phase2


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2-config", type=Path, required=True)
    p.add_argument("--phase2-checkpoint", type=Path, required=True)
    p.add_argument("--patch-index", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase4/data"))
    p.add_argument("--timestamp")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    return p.parse_args()


class LabelSourceDataset(Dataset):
    def __init__(self, rows: list[dict], class_map, ignore: int):
        self.rows = rows
        self.class_map = class_map
        self.ignore = ignore

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_he_patch(
            Path(row["wsi_path"]),
            int(row["x_level0"]),
            int(row["y_level0"]),
            int(row["width_level0"]),
            int(row["height_level0"]),
            int(row["width_10x"]),
        )
        mask = decode_gt_patch(
            Path(row["gt_path"]),
            int(row["x_10x"]),
            int(row["y_10x"]),
            int(row["width_10x"]),
            int(row["height_10x"]),
            self.class_map,
            self.ignore,
        )
        return {
            "patch_id": row["patch_id"],
            "wsi_id": row["wsi_id"],
            "split": row["split"],
            "image": he_to_tensor(image),
            "mask": torch.from_numpy(mask.astype(np.int64)),
        }


def collate(items):
    return {
        "patch_id": [x["patch_id"] for x in items],
        "wsi_id": [x["wsi_id"] for x in items],
        "split": [x["split"] for x in items],
        "image": torch.stack([x["image"] for x in items]),
        "mask": torch.stack([x["mask"] for x in items]),
    }


def main():
    args = parse()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    p2cfg = load_config(args.phase2_config)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"fine_region_labels_{stamp}"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "shards").mkdir()
    if world > 1:
        dist.barrier()

    cache = pd.read_parquet(args.cache_index)
    patches = pd.read_parquet(args.patch_index).set_index("patch_id")
    rows = []
    for rec in cache.to_dict("records"):
        src = patches.loc[rec["patch_id"]]
        rows.append(
            {
                "patch_id": rec["patch_id"],
                "wsi_id": rec["wsi_id"],
                "split": rec["split"],
                "wsi_path": src["wsi_path"],
                "gt_path": src["gt_path"],
                "x_level0": int(src["x_level0"]),
                "y_level0": int(src["y_level0"]),
                "width_level0": int(src["width_level0"]),
                "height_level0": int(src["height_level0"]),
                "x_10x": int(src["x_10x"]),
                "y_10x": int(src["y_10x"]),
                "width_10x": int(src["width_10x"]),
                "height_10x": int(src["height_10x"]),
            }
        )
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[rank::world]

    phase2 = load_phase2(p2cfg, args.phase2_checkpoint, device)
    amp = torch.bfloat16
    ignore = int(p2cfg["data"]["ignore_index"])
    classes = len(p2cfg["data"]["class_map"])
    dataset = LabelSourceDataset(rows, p2cfg["data"]["class_map"], ignore)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate,
    }
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)

    shard_dir = output / "shards" / f"rank{rank:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                out = phase2(image, return_full_assignment=False, return_tokens=True)
                labels = region_labels(out["assignment_low"], mask, classes, ignore)
            labels_np = labels.cpu().numpy().astype(np.int16)
            for i, patch_id in enumerate(batch["patch_id"]):
                path = shard_dir / f"{batch['split'][i]}__{patch_id}.npy"
                np.save(path, labels_np[i])
                records.append(
                    {
                        "patch_id": patch_id,
                        "wsi_id": batch["wsi_id"][i],
                        "split": batch["split"][i],
                        "label_path": str(path),
                        "rank": rank,
                    }
                )
                done += 1
            if done % 64 < args.batch_size:
                rate = done / max(time.time() - t0, 1e-6)
                print({"event": "label_progress", "rank": rank, "done": done, "total": len(rows), "pps": rate}, flush=True)

    pd.DataFrame(records).to_parquet(shard_dir / "index.parquet", index=False)
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = list((output / "shards").glob("rank*/index.parquet"))
        merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if merged["patch_id"].duplicated().any():
            raise RuntimeError("duplicate label patch_id")
        if len(merged) != len(cache if not args.limit else cache.iloc[: args.limit]):
            # when limit set, compare carefully
            expected = args.limit if args.limit else len(cache)
            if len(merged) != expected:
                raise RuntimeError(f"label count {len(merged)} != expected {expected}")
        merged.to_parquet(output / "label_index.parquet", index=False)
        meta = {
            "timestamp": stamp,
            "patch_count": int(len(merged)),
            "splits": merged["split"].value_counts().to_dict(),
            "seconds": time.time() - t0,
            "phase2_checkpoint": str(args.phase2_checkpoint),
            "cache_index": str(args.cache_index),
            "batch_size": args.batch_size,
            "world_size": world,
            "test_used": bool((merged["split"] == "test").any()),
        }
        (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print({"event": "label_complete", "output": str(output), **meta}, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
