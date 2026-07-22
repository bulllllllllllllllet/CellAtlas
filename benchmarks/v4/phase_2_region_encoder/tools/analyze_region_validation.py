from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.dataset import RegionDataset
from benchmarks.v4.phase_2_region_encoder.src.metrics import hard_region_metrics
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation analysis for Phase 2 learned regions.")
    p.add_argument("--config", required=True)
    p.add_argument("--patch-index", required=True)
    p.add_argument("--slic-index", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--timestamp", default=None)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Deterministic prefix of validation patches; omit for all.")
    p.add_argument("--num-cases", type=int, default=20)
    return p.parse_args()


def grid_regions(height: int, width: int, num_regions: int) -> np.ndarray:
    side = int(round(num_regions**0.5))
    yy = np.minimum(np.arange(height) * side // height, side - 1)[:, None]
    xx = np.minimum(np.arange(width) * side // width, side - 1)[None, :]
    return (yy * side + xx).astype(np.uint8)


def per_class_oracle(region: np.ndarray, target: np.ndarray, ignore: int, classes: int) -> tuple[dict[int, float], dict[int, int]]:
    valid = target != ignore
    oracle = np.full(target.shape, ignore, dtype=np.int64)
    for rid in np.unique(region[valid]):
        inside = (region == rid) & valid
        oracle[inside] = int(np.bincount(target[inside], minlength=classes).argmax())
    dice, area = {}, {}
    for cls in range(classes):
        gt = (target == cls) & valid
        count = int(gt.sum())
        if count:
            pred = (oracle == cls) & valid
            dice[cls] = float(2 * (gt & pred).sum() / max(int(gt.sum() + pred.sum()), 1))
            area[cls] = count
    return dice, area


def boundary_ratio(mask: np.ndarray, ignore: int) -> float:
    valid = mask != ignore
    edge = np.zeros_like(valid)
    edge[1:] |= (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    edge[:, 1:] |= (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    return float(edge.sum() / max(valid.sum(), 1))


def palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(20260717)
    return rng.integers(25, 230, size=(n, 3), dtype=np.uint8)


def edge_map(regions: np.ndarray) -> np.ndarray:
    edges = np.zeros(regions.shape, dtype=bool)
    edges[1:] |= regions[1:] != regions[:-1]
    edges[:, 1:] |= regions[:, 1:] != regions[:, :-1]
    return edges


def render_case(image: np.ndarray, target: np.ndarray, regions: np.ndarray, class_rgb: np.ndarray, ignore: int, text: str) -> Image.Image:
    he = Image.fromarray(image)
    gt = np.zeros((*target.shape, 3), dtype=np.uint8)
    valid = target != ignore
    gt[valid] = class_rgb[target[valid]]
    colors = palette(int(regions.max()) + 1)
    reg = colors[regions]
    overlay = image.copy()
    overlay[edge_map(regions)] = (0, 255, 255)
    canvas = Image.new("RGB", (image.shape[1] * 2, image.shape[0] * 2 + 28), "white")
    canvas.paste(he, (0, 28)); canvas.paste(Image.fromarray(gt), (image.shape[1], 28))
    canvas.paste(Image.fromarray(reg), (0, image.shape[0] + 28)); canvas.paste(Image.fromarray(overlay), (image.shape[1], image.shape[0] + 28))
    ImageDraw.Draw(canvas).text((4, 4), text + " | HE / GT / region IDs / region edges", fill="black")
    return canvas


def main() -> None:
    ns = parse(); cfg = load_config(ns.config); stamp = ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(ns.output_root) / f"phase2_validation_analysis_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True); (out / "cases").mkdir()
    device = torch.device(ns.device)
    classes = len(cfg["data"]["class_map"]); ignore = int(cfg["data"]["ignore_index"])
    ds = RegionDataset(ns.patch_index, ns.slic_index, cfg, "val", augment=False)
    if ns.limit is not None: ds.rows, ds.refs = ds.rows[:ns.limit], ds.refs[:ns.limit]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=ns.num_workers, pin_memory=True)
    model = DeepRegionEncoder(cfg, classes, int(cfg["model"]["num_regions"]), int(cfg["model"]["embedding_dim"])).to(device)
    state = torch.load(ns.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True); model.eval()
    class_rgb = np.asarray([item["rgb"] for item in cfg["data"]["class_map"]], dtype=np.uint8)
    records, aggregate = [], defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for index, batch in enumerate(loader):
            output = model(batch["image"].to(device), return_full_assignment=False, return_tokens=False)
            learned = torch.nn.functional.interpolate(output["assignment_low"].argmax(1, keepdim=True).float(), size=batch["mask"].shape[-2:], mode="nearest")[0, 0].to(torch.uint8).cpu().numpy()
            target = batch["mask"][0].numpy(); slic = batch["slic"][0].numpy(); grid = grid_regions(*target.shape, int(cfg["model"]["num_regions"]))
            methods = {"learned": learned, "slic": slic, "grid": grid}
            row = {"index": index, "patch_id": batch["patch_id"][0], "wsi_id": batch["wsi_id"][0], "valid_pixels": int((target != ignore).sum()), "gt_boundary_ratio": boundary_ratio(target, ignore)}
            for name, regions in methods.items():
                metric = hard_region_metrics(regions, target, ignore, classes)
                for key, value in metric.items(): row[f"{name}_{key}"] = value
                dice, area = per_class_oracle(regions, target, ignore, classes)
                for cls, value in dice.items():
                    aggregate[name][cls].append((value, area[cls]))
            records.append(row)
            if (index + 1) % 100 == 0 or index + 1 == len(ds): print({"event": "progress", "done": index + 1, "total": len(ds)}, flush=True)
    frame = pd.DataFrame(records); frame.to_parquet(out / "patch_metrics.parquet", index=False)
    summary = {"input_checkpoint": str(ns.checkpoint), "patch_count": len(frame), "split": "val", "methods": {}}
    for method in ("learned", "slic", "grid"):
        summary["methods"][method] = {key: float(frame[f"{method}_{key}"].mean()) for key in ("region_purity", "boundary_f1", "oracle_region_dice", "active_regions")}
    names = {item["id"]: item["name"] for item in cfg["data"]["class_map"]}
    class_rows = []
    for method, values in aggregate.items():
        for cls, pairs in values.items():
            weighted = float(np.average([x[0] for x in pairs], weights=[x[1] for x in pairs]))
            class_rows.append({"method": method, "class_id": cls, "class_name": names[cls], "oracle_dice_pixel_weighted": weighted, "patches_present": len(pairs), "pixels": int(sum(x[1] for x in pairs))})
    pd.DataFrame(class_rows).to_csv(out / "per_class_oracle_dice.csv", index=False)
    frame["boundary_bin"] = pd.qcut(frame["gt_boundary_ratio"], q=4, duplicates="drop")
    frame["area_bin"] = pd.qcut(frame["valid_pixels"], q=4, duplicates="drop")
    stratified = frame.groupby(["boundary_bin", "area_bin"], observed=True)[["learned_oracle_region_dice", "slic_oracle_region_dice", "grid_oracle_region_dice"]].agg(["mean", "count"])
    stratified.to_csv(out / "failure_strata.csv")
    with (out / "summary.json").open("w") as f: json.dump(summary, f, indent=2)
    rng = random.Random(20260717); ordered = frame.sort_values("learned_oracle_region_dice")
    selections = [("worst", ordered.head(ns.num_cases)), ("best", ordered.tail(ns.num_cases)), ("random", frame.iloc[rng.sample(range(len(frame)), min(ns.num_cases, len(frame)))])]
    for group, selected in selections:
        for _, row in selected.iterrows():
            item = ds[int(row["index"])]
            with torch.no_grad():
                pred = model(item["image"].unsqueeze(0).to(device), return_full_assignment=False, return_tokens=False)["assignment_low"].argmax(1, keepdim=True)
                learned = torch.nn.functional.interpolate(pred.float(), size=item["mask"].shape, mode="nearest")[0, 0].to(torch.uint8).cpu().numpy()
            image = ((item["image"].numpy() * np.asarray([.229, .224, .225])[:, None, None] + np.asarray([.485, .456, .406])[:, None, None]).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            caption = f"{group} | {row['patch_id']} | oracle_dice={row['learned_oracle_region_dice']:.3f} | purity={row['learned_region_purity']:.3f} | boundary_f1={row['learned_boundary_f1']:.3f}"
            render_case(image, item["mask"].numpy(), learned, class_rgb, ignore, caption).save(out / "cases" / f"{group}_{int(row['index']):05d}.png")
    print({"event": "complete", "output": str(out), "summary": summary}, flush=True)


if __name__ == "__main__":
    main()
