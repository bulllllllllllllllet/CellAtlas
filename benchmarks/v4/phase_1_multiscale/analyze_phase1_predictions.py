#!/usr/bin/env python3
"""DDP-safe patch/WSI evaluation plus qualitative failure analysis for Phase 1."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config, save_json, seed_everything
from benchmarks.v4.phase_1_multiscale.src.dataset import SegmentationDataset
from benchmarks.v4.phase_1_multiscale.src.metrics import confusion_matrix, summarize
from benchmarks.v4.phase_1_multiscale.src.model import build_model


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a frozen Phase 1 checkpoint without test-time tuning.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--patch-index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--visuals-per-group", type=int, default=20)
    p.add_argument("--flush-every", type=int, default=128)
    p.add_argument("--max-patches", type=int, help="bounded representative smoke evaluation; omit for the complete split")
    return p.parse_args()


def rank_info() -> tuple[int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def dump_shard(path: Path, rows: list[dict], wsi_conf: dict[str, np.ndarray], global_conf: np.ndarray, complete: bool) -> None:
    save_json(path, {"complete": complete, "rows": rows, "global_confusion": global_conf.tolist(), "wsi_confusion": {k: v.tolist() for k, v in wsi_conf.items()}})


def dice_from_confusion(conf: np.ndarray, background: set[int]) -> tuple[float, list[float]]:
    item = summarize(conf, background)
    return float(item["macro_dice"]), list(map(float, item["per_class_dice"]))


def area_bin(value: float) -> str:
    return "low_<0.50" if value < .5 else "middle_0.50-0.80" if value < .8 else "high_>=0.80"


def boundary_bin(value: float) -> str:
    return "low_<0.02" if value < .02 else "middle_0.02-0.10" if value < .1 else "high_>=0.10"


def summarize_rows(frame: pd.DataFrame, classes: list[dict]) -> dict:
    result: dict[str, object] = {}
    for column in ("tissue_area_bin", "boundary_bin", "sampling_group"):
        result[column] = frame.groupby(column, dropna=False)["patch_macro_dice"].agg(["count", "mean", "median"]).round(6).reset_index().to_dict("records")
    focus = []
    for item in classes:
        class_id = int(item["id"])
        subset = frame[frame["present_classes"].map(lambda raw: class_id in json.loads(raw))]
        class_dice = subset["per_class_dice"].map(lambda raw: json.loads(raw)[class_id]) if len(subset) else pd.Series(dtype=float)
        focus.append({"id": class_id, "name": item["name"], "patch_count": int(len(subset)), "mean_class_dice_when_present": float(class_dice.mean()) if len(subset) else None, "median_class_dice_when_present": float(class_dice.median()) if len(subset) else None})
    result["class_presence_failure_analysis"] = focus
    return result


def palette(config: dict) -> np.ndarray:
    return np.asarray([entry["rgb"] for entry in config["data"]["class_map"]], dtype=np.uint8)


def render(path: Path, sample: dict, prediction: np.ndarray, colors: np.ndarray, ignore: int) -> None:
    image = sample["image"].numpy().transpose(1, 2, 0)
    image = np.clip((image * np.asarray([.229, .224, .225])) + np.asarray([.485, .456, .406]), 0, 1)
    he = (image * 255).astype(np.uint8)
    target = sample["mask"].numpy()
    gt_rgb = np.zeros_like(he); pred_rgb = np.zeros_like(he)
    valid = target != ignore
    gt_rgb[valid] = colors[target[valid]]; pred_rgb[valid] = colors[prediction[valid]]
    overlay = (0.55 * he + 0.45 * pred_rgb).astype(np.uint8)
    Image.fromarray(np.concatenate([he, gt_rgb, pred_rgb, overlay], axis=1)).save(path)


def select_visuals(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    ordered = frame.sort_values(["patch_macro_dice", "patch_id"], kind="stable")
    worst = ordered.head(count).assign(selection="worst")
    best = ordered.tail(count).iloc[::-1].assign(selection="best")
    used = set(worst.patch_id) | set(best.patch_id)
    pool = frame.loc[~frame.patch_id.isin(used)]
    rng = np.random.default_rng(seed)
    random = pool.iloc[rng.choice(len(pool), size=min(count, len(pool)), replace=False)].copy().assign(selection="random")
    return pd.concat([best, worst, random], ignore_index=True)


def main() -> None:
    args = parse(); rank, world = rank_info(); cfg = load_config(args.config)
    if world > 1: dist.init_process_group("nccl"); torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda": raise RuntimeError("Phase 1 evaluation requires an explicit CUDA device")
    seed_everything(int(cfg["project"]["seed"]) + rank)
    if rank == 0:
        if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
        (args.output_dir / "shards").mkdir(parents=True)
        save_json(args.output_dir / "run_metadata.json", {"split": args.split, "checkpoint": str(args.checkpoint), "patch_index": str(args.patch_index), "world_size": world, "num_workers_per_rank": args.num_workers, "visuals_per_group": args.visuals_per_group})
    if world > 1: dist.barrier()
    data = SegmentationDataset(args.patch_index, cfg, args.split)
    if args.max_patches is not None:
        if args.max_patches < 1: raise ValueError("--max-patches must be positive")
        data.rows = data.rows[:args.max_patches]
    rows_by_id = {row["patch_id"]: row for row in data.rows}
    local = Subset(data, list(range(rank, len(data), world)))
    loader_kwargs = {"batch_size": cfg["training"]["batch_size_per_gpu"], "shuffle": False, "num_workers": args.num_workers, "pin_memory": True, "persistent_workers": bool(args.num_workers)}
    if args.num_workers: loader_kwargs.update({"multiprocessing_context": "spawn", "prefetch_factor": 1})
    loader = DataLoader(local, **loader_kwargs)
    model = build_model(cfg, len(cfg["data"]["class_map"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]); model.eval()
    n = len(cfg["data"]["class_map"]); background = set(cfg["data"]["background_class_ids"]); ignore = int(cfg["data"]["ignore_index"])
    records: list[dict] = []; wsi_conf: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((n, n), dtype=np.int64)); total = np.zeros((n, n), dtype=np.int64)
    shard = args.output_dir / "shards" / f"rank_{rank}.json"
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            with torch.autocast(device_type="cuda", enabled=bool(cfg["training"]["amp"])): prediction = model(batch["image"].to(device))["out"].argmax(1).cpu().numpy()
            target = batch["mask"].numpy()
            for offset, patch_id in enumerate(batch["patch_id"]):
                conf = confusion_matrix(target[offset], prediction[offset], n, ignore); patch_dice, per_class = dice_from_confusion(conf, background)
                row = rows_by_id[patch_id]
                records.append({"patch_id": patch_id, "wsi_id": row["wsi_id"], "patch_macro_dice": patch_dice, "per_class_dice": json.dumps(per_class), "tissue_fraction": float(row["tissue_fraction"]), "boundary_fraction": float(row["boundary_fraction"]), "present_classes": row["present_classes"], "sampling_group": row["sampling_group"], "tissue_area_bin": area_bin(float(row["tissue_fraction"])), "boundary_bin": boundary_bin(float(row["boundary_fraction"]))})
                total += conf; wsi_conf[row["wsi_id"]] += conf
            if step % args.flush_every == 0: dump_shard(shard, records, wsi_conf, total, complete=False)
    dump_shard(shard, records, wsi_conf, total, complete=True)
    if world > 1: dist.barrier()
    if rank == 0:
        shards = [json.loads((args.output_dir / "shards" / f"rank_{item}.json").read_text()) for item in range(world)]
        if not all(item["complete"] for item in shards): raise RuntimeError("incomplete evaluation shard")
        frame = pd.DataFrame([row for item in shards for row in item["rows"]]); frame.to_parquet(args.output_dir / "patch_metrics.parquet", index=False)
        global_conf = sum((np.asarray(item["global_confusion"], dtype=np.int64) for item in shards), np.zeros((n, n), dtype=np.int64))
        aggregate: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((n, n), dtype=np.int64))
        for item in shards:
            for wsi_id, conf in item["wsi_confusion"].items(): aggregate[wsi_id] += np.asarray(conf, dtype=np.int64)
        wsi_rows = [{"wsi_id": wsi_id, **summarize(conf, background)} for wsi_id, conf in sorted(aggregate.items())]
        pd.DataFrame(wsi_rows).to_parquet(args.output_dir / "wsi_metrics.parquet", index=False)
        metrics = summarize(global_conf, background); metrics["patch_count"] = int(len(frame)); metrics["wsi_count"] = int(len(wsi_rows)); metrics["wsi_macro_dice_mean"] = float(pd.DataFrame(wsi_rows)["macro_dice"].mean()); metrics["failure_analysis"] = summarize_rows(frame, cfg["data"]["class_map"])
        save_json(args.output_dir / "metrics.json", metrics)
        chosen = select_visuals(frame, args.visuals_per_group, int(cfg["project"]["seed"]))
        visual_dir = args.output_dir / "visuals"; visual_dir.mkdir()
        colors = palette(cfg)
        for position, row in chosen.reset_index(drop=True).iterrows():
            source = data.rows[next(i for i, item in enumerate(data.rows) if item["patch_id"] == row.patch_id)]
            sample = data[data.rows.index(source)]
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=bool(cfg["training"]["amp"])): pred = model(sample["image"].unsqueeze(0).to(device))["out"].argmax(1)[0].cpu().numpy()
            name = f"{row.selection}_{position:02d}_{row.patch_id}.png".replace("/", "_")
            render(visual_dir / name, sample, pred, colors, ignore)
        chosen.to_parquet(args.output_dir / "visualization_manifest.parquet", index=False)
        print(json.dumps({"event": "evaluation_complete", "output": str(args.output_dir), "macro_dice": metrics["macro_dice"], "macro_miou": metrics["macro_miou"], "patch_count": metrics["patch_count"], "wsi_count": metrics["wsi_count"]}), flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
