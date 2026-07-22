#!/usr/bin/env python3
"""Audit point/small/large connected prompt sets against HE and GT."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt

# libvips before torchvision.
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch

import torch

from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor, load_phase2
from benchmarks.v4.phase_5_prompt_encoder.src.prompts import (
    PROMPT_SIZE_SPECS,
    dominant_parent_slots,
    hard_region_adjacency,
    sample_connected_region_set,
)
from benchmarks.v4.phase_5_prompt_encoder.tools.visualize_prompt_episodes import (
    append_jsonl,
    boundary_map,
    gt_rgb,
    make_contact_sheet,
    palette_from_config,
    parse_present_classes,
    region_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase5/data"))
    parser.add_argument("--timestamp")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--class-ids", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--prompt-sizes", nargs="+", choices=tuple(PROMPT_SIZE_SPECS), default=list(PROMPT_SIZE_SPECS))
    parser.add_argument("--min-purity", type=float, default=0.90)
    parser.add_argument("--min-valid-region-pixels", type=int, default=1024)
    parser.add_argument("--max-candidates-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def prompt_anchor(mask: np.ndarray) -> tuple[int, int]:
    if not mask.any():
        raise ValueError("empty prompt mask")
    distance = distance_transform_edt(mask)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(x), int(y)


def draw_marks(ax, positive_anchor: tuple[int, int], negatives: list[tuple[int, int]]) -> None:
    x, y = positive_anchor
    ax.scatter(x, y, s=180, marker="o", facecolors="none", edgecolors="lime", linewidths=3, zorder=6)
    ax.text(x + 6, y - 7, "+ region", color="lime", fontsize=8, weight="bold", zorder=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})
    for index, (nx, ny) in enumerate(negatives, start=1):
        ax.scatter(nx, ny, s=145, marker="X", c="red", edgecolors="white", linewidths=1.5, zorder=6)
        ax.text(nx + 6, ny - 7, f"- {index}", color="red", fontsize=8, weight="bold", zorder=7,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})


def prompt_overlay(image: np.ndarray, positive: np.ndarray, negative: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[positive] = 0.40 * out[positive] + 0.60 * np.asarray([0, 255, 0])
    out[negative] = 0.40 * out[negative] + 0.60 * np.asarray([255, 0, 0])
    out[boundary] = np.asarray([0, 255, 255])
    return out.clip(0, 255).astype(np.uint8)


def comparison_rgb(gt_target: np.ndarray, prompt_positive: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt_target.shape, 3), dtype=np.uint8)
    only_gt = gt_target & ~prompt_positive
    only_prompt = prompt_positive & ~gt_target
    overlap = gt_target & prompt_positive
    out[only_gt] = np.asarray([0, 180, 0], dtype=np.uint8)
    out[only_prompt] = np.asarray([0, 210, 210], dtype=np.uint8)
    out[overlap] = np.asarray([255, 230, 0], dtype=np.uint8)
    return out


def save_panel(path: Path, quicklook: Path, image: np.ndarray, mask: np.ndarray, hard: np.ndarray,
               target: int, target_name: str, size_name: str, positive_slots: np.ndarray,
               negative_slots: np.ndarray, positive_anchor: tuple[int, int], negative_anchors: list[tuple[int, int]],
               palette: np.ndarray, ignore: int, hierarchy: str, coverage: float, purity: float) -> None:
    positive = np.isin(hard, positive_slots)
    negative = np.isin(hard, negative_slots)
    boundary = boundary_map(hard)
    overlay = prompt_overlay(image, positive, negative, boundary)
    gt_target = mask == target
    compare = comparison_rgb(gt_target, positive)
    binary = np.zeros((*mask.shape, 3), dtype=np.uint8)
    binary[gt_target] = np.asarray([0, 220, 0], dtype=np.uint8)
    binary[mask == ignore] = np.asarray([180, 0, 180], dtype=np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    axes[0].imshow(image); axes[0].set_title("HE (10x)")
    axes[1].imshow(gt_rgb(mask, palette, ignore)); axes[1].set_title("12-class GT")
    axes[2].imshow(binary); axes[2].set_title(f"Binary GT: {target_name}")
    axes[3].imshow(compare); axes[3].set_title("GT only green | prompt only cyan | overlap yellow")
    axes[4].imshow(overlay); draw_marks(axes[4], positive_anchor, negative_anchors)
    axes[4].set_title("HE + positive region + hard negatives")
    axes[5].imshow(overlay); draw_marks(axes[5], positive_anchor, negative_anchors)
    axes[5].set_title(f"{hierarchy}\ncoverage={coverage*100:.2f}% purity={purity:.3f}")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Class {target}: {target_name} | {size_name} prompt", fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(overlay); draw_marks(axes[0], positive_anchor, negative_anchors); axes[0].set_title("HE + prompt")
    axes[1].imshow(compare); draw_marks(axes[1], positive_anchor, negative_anchors); axes[1].set_title("GT vs prompt")
    for ax in axes: ax.axis("off")
    fig.tight_layout(pad=0.2); fig.savefig(quicklook, dpi=105); plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.phase2_config)
    classes = cfg["data"]["class_map"]
    names = {int(item["id"]): str(item["name"]) for item in classes}
    unknown = sorted(set(args.class_ids) - set(names))
    if unknown: raise ValueError(f"unknown class ids: {unknown}")
    ignore = int(cfg["data"]["ignore_index"])
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"prompt_size_visualization_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("panels", "quicklooks"):
        for size in args.prompt_sizes:
            (output / name / size).mkdir(parents=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    model = load_phase2(cfg, args.phase2_checkpoint, device)
    cache = pd.read_parquet(args.cache_index)
    cache = cache[cache["split"] == args.split].set_index("patch_id", drop=False)
    label_index = pd.read_parquet(args.label_index)
    label_index = label_index[label_index["split"] == args.split].set_index("patch_id", drop=False)
    patches = pd.read_parquet(args.patch_index)
    patches = patches[patches["split"] == args.split].set_index("patch_id", drop=False)
    common = cache.index.intersection(label_index.index).intersection(patches.index)
    if len(common) != len(cache): raise RuntimeError("cache/label/patch index mismatch")
    rng = np.random.default_rng(args.seed)
    order = common.to_numpy().copy(); rng.shuffle(order)
    palette = palette_from_config(cfg)
    completed = output / "completed.jsonl"
    records = []

    for size_name in args.prompt_sizes:
        spec = PROMPT_SIZE_SPECS[size_name]
        for target in args.class_ids:
            candidates = []
            for patch_id in order:
                if target not in parse_present_classes(patches.loc[patch_id, "present_classes"]): continue
                labels = np.load(label_index.loc[patch_id, "label_path"])
                if int((labels == target).sum()) >= int(spec["min_slots"]) and np.any((labels != target) & (labels != ignore)):
                    candidates.append(patch_id)
                if len(candidates) >= args.max_candidates_per_class: break
            print(json.dumps({"event":"size_class_candidates","size":size_name,"class":target,"count":len(candidates)}), flush=True)
            selected = None
            for attempt, patch_id in enumerate(candidates, start=1):
                row = patches.loc[patch_id].to_dict()
                with np.load(cache.loc[patch_id, "shard_path"]) as archive:
                    z = {key: archive[key] for key in archive.files}
                labels = np.load(label_index.loc[patch_id, "label_path"]).astype(np.int16)
                active = z["fine_active"].astype(bool)
                image = read_he_patch(Path(row["wsi_path"]), int(row["x_level0"]), int(row["y_level0"]),
                                      int(row["width_level0"]), int(row["height_level0"]), 512)
                mask = decode_gt_patch(Path(row["gt_path"]), int(row["x_10x"]), int(row["y_10x"]),
                                       int(row["width_10x"]), int(row["height_10x"]), classes, ignore)
                tensor = he_to_tensor(image).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                    result = model(tensor, return_full_assignment=True, return_tokens=True)
                recomputed = region_labels(result["assignment_low"], torch.from_numpy(mask).unsqueeze(0).to(device),
                                           len(classes), ignore)[0].cpu().numpy().astype(np.int16)
                assignment = result["assignment"][0].float().cpu().numpy()
                hard = assignment.argmax(0).astype(np.int16)
                purity, valid_pixels = region_statistics(hard, mask, len(classes), ignore, labels.size)
                total_pixels = np.bincount(hard.ravel(), minlength=labels.size)
                stable_clean = active & (recomputed == labels) & (purity >= args.min_purity) & (valid_pixels >= args.min_valid_region_pixels)
                eligible = stable_clean & (labels == target)
                adjacency = hard_region_adjacency(hard, labels.size)
                try:
                    positive_slots = sample_connected_region_set(
                        adjacency, eligible, total_pixels, rng, min_slots=int(spec["min_slots"]),
                        max_slots=int(spec["max_slots"]), min_fraction=float(spec["min_fraction"]),
                        max_fraction=float(spec["max_fraction"]), patch_pixels=hard.size)
                except ValueError:
                    continue
                negative_pool = np.flatnonzero(stable_clean & (labels != target) & (labels != ignore))
                if negative_pool.size < int(spec["negative_slots"]): continue
                px = z["fine_centroid_x"]; py = z["fine_centroid_y"]
                center = np.asarray([px[positive_slots].mean(), py[positive_slots].mean()])
                distance = (px[negative_pool]-center[0])**2 + (py[negative_pool]-center[1])**2
                negative_slots = negative_pool[np.argsort(distance)[:int(spec["negative_slots"])]].astype(np.int64)
                positive_mask = np.isin(hard, positive_slots)
                positive_anchor = prompt_anchor(positive_mask & (mask == target))
                negative_anchors = [prompt_anchor((hard == slot) & (mask == labels[slot])) for slot in negative_slots]
                coverage = float(positive_mask.mean())
                valid_union = positive_mask & (mask != ignore)
                set_purity = float(((mask == target) & positive_mask).sum() / max(valid_union.sum(), 1))
                middle = dominant_parent_slots(positive_slots, z["fine_middle_edge_index"], z["fine_middle_edge_weight"])
                coarse = dominant_parent_slots(middle, z["middle_coarse_edge_index"], z["middle_coarse_edge_weight"])
                hierarchy = f"fine {len(positive_slots)} -> middle {len(middle)} -> coarse {len(coarse)} dominant parents"
                selected = {
                    "patch_id":patch_id,"wsi_id":row["wsi_id"],"split":args.split,"target_class":target,
                    "target_name":names[target],"prompt_size":size_name,"candidate_attempt":attempt,
                    "positive_slots":positive_slots.tolist(),"negative_slots":negative_slots.tolist(),
                    "positive_slot_count":len(positive_slots),"negative_slot_count":len(negative_slots),
                    "coverage_fraction":coverage,"set_purity":set_purity,"middle_parent_slots":middle.tolist(),
                    "coarse_parent_slots":coarse.tolist(),"connected":True,
                }
                stem = f"{size_name}__class_{target:02d}_{names[target]}__{patch_id}"
                panel = output/"panels"/size_name/f"{stem}.png"; quick = output/"quicklooks"/size_name/f"{stem}.png"
                save_panel(panel, quick, image, mask, hard, target, names[target], size_name, positive_slots,
                           negative_slots, positive_anchor, negative_anchors, palette, ignore, hierarchy, coverage, set_purity)
                selected["panel_path"] = str(panel); selected["quicklook_path"] = str(quick)
                (output/"panels"/size_name/f"{stem}.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
                append_jsonl(completed, selected); records.append(selected); break
            if selected is None:
                raise RuntimeError(f"no {size_name} prompt for class {target} passed contract after {len(candidates)} candidates")

    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in records]).to_parquet(output/"prompt_size_index.parquet",index=False)
    for size_name in args.prompt_sizes:
        make_contact_sheet([r for r in records if r["prompt_size"]==size_name], output/f"contact_sheet_{size_name}.png")
    meta = {"timestamp":stamp,"purpose":"pre-training human audit only","split":args.split,"test_used":False,
            "episode_count":len(records),"episodes_per_size":{s:sum(r["prompt_size"]==s for r in records) for s in args.prompt_sizes},
            "size_specs":PROMPT_SIZE_SPECS,"min_purity":args.min_purity,"min_valid_region_pixels":args.min_valid_region_pixels,
            "multilevel_policy":"record dominant fine->middle->coarse parents; Phase5 backbone remains fine-only",
            "validation":{"all_connected":all(r["connected"] for r in records),
                          "minimum_set_purity":min(r["set_purity"] for r in records),
                          "coverage_by_size":{s:{"min":min(r["coverage_fraction"] for r in records if r["prompt_size"]==s),
                                                  "max":max(r["coverage_fraction"] for r in records if r["prompt_size"]==s)} for s in args.prompt_sizes}}}
    (output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"event":"prompt_size_visualization_complete","output":str(output),**meta["validation"]}))


if __name__ == "__main__":
    main()
