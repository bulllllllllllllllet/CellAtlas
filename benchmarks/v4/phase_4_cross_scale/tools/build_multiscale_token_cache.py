#!/usr/bin/env python3
"""Build multi-scale frozen token + sparse edge cache (batched pilot/formal path)."""
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
from torch.utils.data import DataLoader

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch  # noqa: F401  libvips before torchvision
from benchmarks.v4.phase_4_cross_scale.src.cache_dataset import (
    MultiscaleCacheDataset,
    balance_by_wsi,
    collate_multiscale_cache,
)
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import (
    encode_scale_batch,
    fuse_fine_cells_batch,
    load_cell_module,
    load_phase2,
)
from benchmarks.v4.phase_4_cross_scale.src.geometry import (
    assignment_centroids_areas,
    edge_invariants,
    parent_child_edges,
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase3-config", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--slic-index", type=Path, required=True)
    parser.add_argument("--train-routing", type=Path)
    parser.add_argument("--val-routing", type=Path)
    parser.add_argument("--test-routing", type=Path)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument(
        "--stratified-pilot",
        action="store_true",
        help="sample interior/boundary/rare/hard-negative proportionally",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase4/data"))
    parser.add_argument("--timestamp")
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def select_rows(df: pd.DataFrame, limit: int | None, stratified: bool, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    if not stratified:
        return df.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)
    groups = df.groupby("sampling_group", sort=True)
    rng = np.random.default_rng(seed)
    allocation = {}
    total = len(df)
    for name, part in groups:
        allocation[name] = min(len(part), int(np.floor(limit * len(part) / total)))
    remaining = limit - sum(allocation.values())
    order = sorted(
        allocation,
        key=lambda n: (limit * len(groups.get_group(n)) / total - allocation[n], len(groups.get_group(n))),
        reverse=True,
    )
    while remaining:
        progressed = False
        for name in order:
            if allocation[name] < len(groups.get_group(name)):
                allocation[name] += 1
                remaining -= 1
                progressed = True
            if not remaining:
                break
        if not progressed:
            break
    chosen = []
    for name, part in groups:
        ids = part.index.to_numpy()
        take = allocation[name]
        chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
    return df.loc[sorted(chosen)].reset_index(drop=True)


def save_patch(path: Path, packed: dict):
    np.savez(
        path,
        fine_tokens=packed["fine_tokens"].astype(np.float16),
        middle_tokens=packed["middle_tokens"].astype(np.float16),
        coarse_tokens=packed["coarse_tokens"].astype(np.float16),
        fine_active=packed["fine_active"],
        middle_active=packed["middle_active"],
        coarse_active=packed["coarse_active"],
        fine_centroid_x=packed["fine_centroid_x"].astype(np.float32),
        fine_centroid_y=packed["fine_centroid_y"].astype(np.float32),
        middle_centroid_x=packed["middle_centroid_x"].astype(np.float32),
        middle_centroid_y=packed["middle_centroid_y"].astype(np.float32),
        coarse_centroid_x=packed["coarse_centroid_x"].astype(np.float32),
        coarse_centroid_y=packed["coarse_centroid_y"].astype(np.float32),
        fine_area=packed["fine_area"].astype(np.float32),
        middle_area=packed["middle_area"].astype(np.float32),
        coarse_area=packed["coarse_area"].astype(np.float32),
        fine_middle_edge_index=packed["fine_middle_edge_index"],
        fine_middle_edge_weight=packed["fine_middle_edge_weight"],
        middle_coarse_edge_index=packed["middle_coarse_edge_index"],
        middle_coarse_edge_weight=packed["middle_coarse_edge_weight"],
    )


def process_batch(batch, phase2, cell, device, amp, top_k: int) -> list[dict]:
    fine = fuse_fine_cells_batch(
        phase2,
        cell,
        batch["image_10x"],
        batch["cells"],
        batch["cell_valid"],
        batch["total_cell_count"].float(),
        device,
        amp,
    )
    middle = encode_scale_batch(phase2, batch["image_5x"], device, amp)
    coarse = encode_scale_batch(phase2, batch["image_2p5x"], device, amp)
    outputs = []
    batch_size = len(batch["patch_id"])
    # Use backbone-resolution soft assignment for geometry/edges. Full-res
    # assignment is only needed for pixel overlays and is ~16x denser.
    for i in range(batch_size):
        fine_asg = fine["assignment_low"][i]
        mid_asg = middle["assignment_low"][i]
        coarse_asg = coarse["assignment_low"][i]
        fine_geom = assignment_centroids_areas(fine_asg, *batch["box_10x"][i])
        mid_geom = assignment_centroids_areas(mid_asg, *batch["box_5x"][i])
        coarse_geom = assignment_centroids_areas(coarse_asg, *batch["box_2p5x"][i])
        fine_middle = parent_child_edges(
            fine_asg,
            batch["box_10x"][i],
            mid_asg,
            batch["box_5x"][i],
            top_k=top_k,
        )
        middle_coarse = parent_child_edges(
            mid_asg,
            batch["box_5x"][i],
            coarse_asg,
            batch["box_2p5x"][i],
            top_k=top_k,
        )
        inv_fm = edge_invariants(fine_middle)
        inv_mc = edge_invariants(middle_coarse)
        if not (inv_fm["passed"] and inv_mc["passed"]):
            raise RuntimeError(
                f"edge invariant failed patch={batch['patch_id'][i]} fm={inv_fm} mc={inv_mc}"
            )
        if not (
            np.isfinite(fine["fused_tokens"][i]).all()
            and np.isfinite(middle["tokens"][i]).all()
            and np.isfinite(coarse["tokens"][i]).all()
        ):
            raise FloatingPointError(f"non-finite tokens patch={batch['patch_id'][i]}")
        outputs.append(
            {
                "patch_id": batch["patch_id"][i],
                "wsi_id": batch["wsi_id"][i],
                "split": batch["split"][i],
                "sampling_group": batch["sampling_group"][i],
                "fine_tokens": fine["fused_tokens"][i],
                "middle_tokens": middle["tokens"][i],
                "coarse_tokens": coarse["tokens"][i],
                "fine_active": fine_geom["active"],
                "middle_active": mid_geom["active"],
                "coarse_active": coarse_geom["active"],
                "fine_centroid_x": fine_geom["centroid_x"],
                "fine_centroid_y": fine_geom["centroid_y"],
                "middle_centroid_x": mid_geom["centroid_x"],
                "middle_centroid_y": mid_geom["centroid_y"],
                "coarse_centroid_x": coarse_geom["centroid_x"],
                "coarse_centroid_y": coarse_geom["centroid_y"],
                "fine_area": fine_geom["area"],
                "middle_area": mid_geom["area"],
                "coarse_area": coarse_geom["area"],
                "fine_middle_edge_index": fine_middle["edge_index"],
                "fine_middle_edge_weight": fine_middle["edge_weight"],
                "middle_coarse_edge_index": middle_coarse["edge_index"],
                "middle_coarse_edge_weight": middle_coarse["edge_weight"],
            }
        )
    return outputs


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
    p3cfg = load_config(args.phase3_config)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"multiscale_token_cache_{stamp}"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "shards").mkdir()
    if world > 1:
        dist.barrier()

    phase2 = load_phase2(p2cfg, args.phase2_checkpoint, device)
    cell = load_cell_module(
        int(p3cfg["model"]["region_dim"]),
        int(p3cfg["data"]["cell_feature_dim"]),
        args.cell_checkpoint,
        device,
    )
    amp = torch.bfloat16 if p3cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    max_cells = int(p3cfg["data"]["max_cells_per_patch"])

    routing = {"train": args.train_routing, "val": args.val_routing, "test": args.test_routing}
    all_index = pd.read_parquet(args.patch_index)
    items: list[dict] = []
    for split in args.splits.split(","):
        split = split.strip()
        part = all_index[all_index["split"] == split].reset_index(drop=True)
        if routing[split] is None:
            raise ValueError(f"--{split}-routing required")
        route = pd.read_parquet(routing[split]).sort_values("source_index").reset_index(drop=True)
        if len(route) != len(part):
            raise ValueError(f"{split} routing/patch mismatch {len(route)} != {len(part)}")
        part = part.copy()
        part["source_index"] = np.arange(len(part))
        selected = select_rows(
            part,
            args.limit_per_split,
            args.stratified_pilot,
            args.seed + {"train": 0, "val": 1, "test": 2}[split],
        )
        for _, row in selected.iterrows():
            route_row = route.iloc[int(row["source_index"])].to_dict()
            items.append({"split": split, "row": row.to_dict(), "route": route_row})

    work = balance_by_wsi(items, rank, world)
    dataset = MultiscaleCacheDataset(work, max_cells=max_cells)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": lambda batch: collate_multiscale_cache(batch, max_cells),
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)

    shard_dir = output / "shards" / f"rank{rank:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_path = shard_dir / "completed.jsonl"
    failures_path = shard_dir / "failures.jsonl"
    completed_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")
    records = []
    t0 = time.time()
    done = 0
    for batch in loader:
        try:
            packed_list = process_batch(batch, phase2, cell, device, amp, args.top_k)
            for packed in packed_list:
                path = shard_dir / f"{packed['split']}__{packed['patch_id']}.npz"
                save_patch(path, packed)
                rec = {
                    "patch_id": packed["patch_id"],
                    "wsi_id": packed["wsi_id"],
                    "split": packed["split"],
                    "sampling_group": packed["sampling_group"],
                    "shard_path": str(path),
                    "rank": rank,
                }
                records.append(rec)
                with completed_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec) + "\n")
                done += 1
        except Exception as exc:  # noqa: BLE001
            fail = {
                "patch_ids": batch["patch_id"],
                "split": batch["split"],
                "rank": rank,
                "error": repr(exc),
            }
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fail) + "\n")
            raise
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        print(
            {
                "event": "cache_progress",
                "rank": rank,
                "done": done,
                "total": len(work),
                "patches_per_sec": rate,
                "batch_size": args.batch_size,
            },
            flush=True,
        )

    if records:
        pd.DataFrame(records).to_parquet(shard_dir / "index.parquet", index=False)
    rank_seconds = time.time() - t0
    rank_stats = torch.tensor(
        [float(done), float(rank_seconds), float(done / max(rank_seconds, 1e-6))],
        device=device,
    )
    stats_list = [torch.zeros_like(rank_stats) for _ in range(world)] if world > 1 and rank == 0 else None
    if world > 1:
        dist.gather(rank_stats, stats_list if rank == 0 else None, dst=0)
        dist.barrier()
    else:
        stats_list = [rank_stats.cpu()]

    if rank == 0:
        parts = list((output / "shards").glob("rank*/index.parquet"))
        if not parts:
            raise RuntimeError("no shard indexes written")
        merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if merged["patch_id"].duplicated().any():
            raise RuntimeError("duplicate patch_id in cache")
        merged.to_parquet(output / "cache_index.parquet", index=False)
        completed_lines = []
        failure_lines = []
        for path in sorted((output / "shards").glob("rank*/completed.jsonl")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                completed_lines.extend(text.splitlines())
        for path in sorted((output / "shards").glob("rank*/failures.jsonl")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                failure_lines.extend(text.splitlines())
        (output / "completed.jsonl").write_text(
            "\n".join(completed_lines) + ("\n" if completed_lines else ""), encoding="utf-8"
        )
        (output / "failures.jsonl").write_text(
            "\n".join(failure_lines) + ("\n" if failure_lines else ""), encoding="utf-8"
        )
        per_rank = []
        for i, tensor in enumerate(stats_list or []):
            values = tensor.detach().cpu().tolist()
            per_rank.append(
                {
                    "rank": i,
                    "patches": int(values[0]),
                    "seconds": float(values[1]),
                    "patches_per_sec": float(values[2]),
                }
            )
        wall = max((row["seconds"] for row in per_rank), default=rank_seconds)
        global_rate = float(len(merged) / max(wall, 1e-6))
        meta = {
            "timestamp": stamp,
            "patch_count": int(len(merged)),
            "splits": merged["split"].value_counts().to_dict(),
            "sampling_groups": merged["sampling_group"].value_counts().to_dict()
            if "sampling_group" in merged
            else {},
            "world_size": world,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "limit_per_split": args.limit_per_split,
            "stratified_pilot": bool(args.stratified_pilot),
            "phase2_checkpoint": str(args.phase2_checkpoint),
            "cell_checkpoint": str(args.cell_checkpoint),
            "failure_lines": len(failure_lines),
            "seconds_wall": wall,
            "global_patches_per_sec": global_rate,
            "per_rank": per_rank,
            "builder": "batched_v2",
            "test_used": bool((merged["split"] == "test").any()),
            "est_formal_hours_at_this_rate": float(90744 / max(global_rate, 1e-6) / 3600),
        }
        (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print({"event": "cache_complete", "output": str(output), **meta}, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
