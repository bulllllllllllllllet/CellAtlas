#!/usr/bin/env python3
"""Generate class-balanced HE/GT/prompt audit panels before Phase-5 training."""
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

# libvips must be imported before torchvision through DeepRegionEncoder.
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch

import torch

from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor, load_phase2
from benchmarks.v4.phase_4_cross_scale.src.geometry import assignment_centroids_areas
from benchmarks.v4.phase_5_prompt_encoder.src.prompts import sample_region_prompt_episode


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
    parser.add_argument("--patch-id", help="restrict a single-class diagnostic to one patch")
    parser.add_argument("--min-purity", type=float, default=0.90)
    parser.add_argument("--min-region-pixels", type=int, default=32)
    parser.add_argument("--max-candidates-per-class", type=int, default=80)
    parser.add_argument("--max-positive", type=int, default=3)
    parser.add_argument("--max-negative", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_present_classes(value) -> set[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return {int(item) for item in value}


def region_statistics(
    hard: np.ndarray, mask: np.ndarray, classes: int, ignore: int, slots: int
) -> tuple[np.ndarray, np.ndarray]:
    votes = np.zeros((slots, classes), dtype=np.int64)
    valid = mask != ignore
    np.add.at(votes, (hard[valid], mask[valid]), 1)
    totals = votes.sum(1)
    purity = np.divide(votes.max(1), totals, out=np.zeros(slots, dtype=np.float64), where=totals > 0)
    return purity.astype(np.float32), totals


def interior_point(hard: np.ndarray, mask: np.ndarray, slot: int, class_id: int) -> tuple[int, int]:
    candidate = (hard == slot) & (mask == class_id)
    if not candidate.any():
        raise RuntimeError(f"slot {slot} has no pixel of class {class_id}")
    distance = distance_transform_edt(candidate)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    if int(hard[y, x]) != slot or int(mask[y, x]) != class_id:
        raise RuntimeError("interior prompt point failed slot/class invariant")
    return int(x), int(y)


def boundary_map(hard: np.ndarray) -> np.ndarray:
    boundary = np.zeros(hard.shape, dtype=bool)
    boundary[1:] |= hard[1:] != hard[:-1]
    boundary[:, 1:] |= hard[:, 1:] != hard[:, :-1]
    return boundary


def palette_from_config(config: dict) -> np.ndarray:
    return np.asarray([item["rgb"] for item in config["data"]["class_map"]], dtype=np.uint8)


def gt_rgb(mask: np.ndarray, palette: np.ndarray, ignore: int) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore
    rgb[valid] = palette[mask[valid]]
    rgb[~valid] = np.asarray([32, 32, 32], dtype=np.uint8)
    return rgb


def add_prompts(ax, clicks: list[dict]) -> None:
    for click in clicks:
        positive = click["sign"] == "positive"
        ax.scatter(
            click["x"],
            click["y"],
            s=150,
            marker="o" if positive else "X",
            facecolors="none" if positive else "red",
            edgecolors="lime" if positive else "white",
            linewidths=2.5,
            zorder=5,
        )
        ax.text(
            click["x"] + 7,
            click["y"] - 7,
            f"{'+' if positive else '-'}s{click['slot']}",
            color="lime" if positive else "red",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
            zorder=6,
        )


def save_panel(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    hard: np.ndarray,
    labels: np.ndarray,
    target: int,
    target_name: str,
    clicks: list[dict],
    palette: np.ndarray,
    ignore: int,
) -> None:
    gt = gt_rgb(mask, palette, ignore)
    binary = np.zeros((*mask.shape, 3), dtype=np.uint8)
    binary[mask == target] = np.asarray([0, 220, 0], dtype=np.uint8)
    binary[mask == ignore] = np.asarray([180, 0, 180], dtype=np.uint8)
    boundary = boundary_map(hard)

    region_overlay = image.astype(np.float32).copy()
    selected_positive = np.zeros(mask.shape, dtype=bool)
    selected_negative = np.zeros(mask.shape, dtype=bool)
    for click in clicks:
        selected = hard == int(click["slot"])
        if click["sign"] == "positive":
            selected_positive |= selected
        else:
            selected_negative |= selected
    region_overlay[selected_positive] = 0.45 * region_overlay[selected_positive] + 0.55 * np.asarray([0, 255, 0])
    region_overlay[selected_negative] = 0.45 * region_overlay[selected_negative] + 0.55 * np.asarray([255, 0, 0])
    region_overlay[boundary] = np.asarray([0, 255, 255])
    region_overlay = region_overlay.clip(0, 255).astype(np.uint8)

    projected = np.zeros((*mask.shape, 3), dtype=np.uint8)
    projected[np.isin(hard, np.flatnonzero(labels == target))] = np.asarray([0, 220, 220], dtype=np.uint8)
    projected[mask == target] = np.maximum(projected[mask == target], np.asarray([0, 180, 0], dtype=np.uint8))
    projected[boundary] = np.asarray([255, 255, 255], dtype=np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title("HE (10x, 512x512)")
    axes[1].imshow(gt)
    axes[1].set_title("12-class GT (ignore=dark gray)")
    axes[2].imshow(binary)
    axes[2].set_title(f"Binary GT: {target_name} (green)")
    axes[3].imshow(projected)
    axes[3].set_title("GT target (green) vs region target (cyan)")
    axes[4].imshow(image)
    add_prompts(axes[4], clicks)
    axes[4].set_title("Raw visual prompts: + green circle / - red X")
    axes[5].imshow(region_overlay)
    add_prompts(axes[5], clicks)
    axes[5].set_title("Prompted regions: + green / - red; boundaries cyan")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Target {target}: {target_name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_contact_sheet(items: list[dict], output: Path) -> None:
    rows = max(1, int(np.ceil(len(items) / 3)))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 3.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items, strict=False):
        with Image.open(item["quicklook_path"]) as tile:
            ax.imshow(tile.convert("RGB"))
        ax.set_title(f"Class {item['target_class']}: {item['target_name']}", fontsize=13, weight="bold")
    fig.suptitle("Phase 5 clean visual prompts: HE + prompts | target GT + prompts", fontsize=17)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)



def save_quicklook(path: Path, image: np.ndarray, mask: np.ndarray, target: int, clicks: list[dict]) -> None:
    target_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    target_rgb[mask == target] = np.asarray([0, 220, 0], dtype=np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    add_prompts(axes[0], clicks)
    axes[0].set_title("HE + prompts")
    axes[1].imshow(target_rgb)
    add_prompts(axes[1], clicks)
    axes[1].set_title("Target GT + prompts")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.phase2_config)
    ignore = int(config["data"]["ignore_index"])
    class_items = config["data"]["class_map"]
    class_names = {int(item["id"]): str(item["name"]) for item in class_items}
    class_ids = [int(value) for value in args.class_ids]
    unknown = sorted(set(class_ids) - set(class_names))
    if unknown:
        raise ValueError(f"unknown class ids: {unknown}")
    if args.patch_id is not None and len(class_ids) != 1:
        raise ValueError("--patch-id requires exactly one --class-ids value")
    if not 0.5 < args.min_purity <= 1.0:
        raise ValueError("min-purity must be in (0.5,1.0]")

    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"prompt_visualization_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    panels = output / "panels"
    quicklooks = output / "quicklooks"
    panels.mkdir()
    quicklooks.mkdir()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        torch.cuda.set_device(device)
    model = load_phase2(config, args.phase2_checkpoint, device)

    cache = pd.read_parquet(args.cache_index)
    cache = cache[cache["split"] == args.split].set_index("patch_id", drop=False)
    label_index = pd.read_parquet(args.label_index)
    label_index = label_index[label_index["split"] == args.split].set_index("patch_id", drop=False)
    patch_index = pd.read_parquet(args.patch_index)
    patch_index = patch_index[patch_index["split"] == args.split].set_index("patch_id", drop=False)
    common = cache.index.intersection(label_index.index).intersection(patch_index.index)
    if len(common) != len(cache):
        raise RuntimeError(f"split index mismatch: cache={len(cache)} common={len(common)}")

    palette = palette_from_config(config)
    rng = np.random.default_rng(args.seed)
    candidate_order = common.to_numpy().copy()
    rng.shuffle(candidate_order)
    if args.patch_id is not None:
        if args.patch_id not in common:
            raise ValueError(f"patch-id is not in the {args.split} cache: {args.patch_id}")
        candidate_order = np.asarray([args.patch_id], dtype=object)
    completed = output / "completed.jsonl"
    records: list[dict] = []

    for class_id in class_ids:
        eligible_patch_ids = []
        for patch_id in candidate_order:
            # The patch manifest is one in-memory table. Use it to avoid
            # thousands of random NFS opens for rare classes.
            if class_id not in parse_present_classes(patch_index.loc[patch_id, "present_classes"]):
                continue
            labels = np.load(label_index.loc[patch_id, "label_path"]).astype(np.int16)
            valid_labels = labels[labels != ignore]
            if np.any(valid_labels == class_id) and np.any(valid_labels != class_id):
                eligible_patch_ids.append(patch_id)
            if len(eligible_patch_ids) >= args.max_candidates_per_class:
                break
        if not eligible_patch_ids:
            raise RuntimeError(f"no candidate patch for class {class_id}")

        selected = None
        for attempt, patch_id in enumerate(eligible_patch_ids, start=1):
            if attempt == 1 or attempt % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "prompt_candidate",
                            "class_id": class_id,
                            "attempt": attempt,
                            "candidate_count": len(eligible_patch_ids),
                        }
                    ),
                    flush=True,
                )
            patch = patch_index.loc[patch_id].to_dict()
            cache_row = cache.loc[patch_id]
            label_row = label_index.loc[patch_id]
            with np.load(cache_row["shard_path"]) as archive:
                z = {key: archive[key] for key in archive.files}
            labels = np.load(label_row["label_path"]).astype(np.int16)
            active = z["fine_active"].astype(bool)
            image = read_he_patch(
                Path(patch["wsi_path"]),
                int(patch["x_level0"]),
                int(patch["y_level0"]),
                int(patch["width_level0"]),
                int(patch["height_level0"]),
                512,
            )
            mask = decode_gt_patch(
                Path(patch["gt_path"]),
                int(patch["x_10x"]),
                int(patch["y_10x"]),
                int(patch["width_10x"]),
                int(patch["height_10x"]),
                class_items,
                ignore,
            )
            tensor = he_to_tensor(image).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                result = model(tensor, return_full_assignment=True, return_tokens=True)
            recomputed = region_labels(
                result["assignment_low"],
                torch.from_numpy(mask).unsqueeze(0).to(device),
                len(class_items),
                ignore,
            )[0].cpu().numpy().astype(np.int16)
            assignment = result["assignment"][0].float().cpu().numpy()
            hard = assignment.argmax(0).astype(np.int16)
            purity, region_pixels = region_statistics(hard, mask, len(class_items), ignore, labels.size)
            label_mismatch = np.flatnonzero(recomputed != labels)
            label_mismatch_detail = [
                    {
                        "slot": int(slot),
                        "cached": int(labels[slot]),
                        "recomputed": int(recomputed[slot]),
                        "active": bool(active[slot]),
                        "online_purity": float(purity[slot]),
                        "online_valid_pixels": int(region_pixels[slot]),
                    }
                    for slot in label_mismatch
            ]
            high_purity = (purity >= args.min_purity) & (region_pixels >= args.min_region_pixels)
            unstable_prompt_candidates = label_mismatch[
                active[label_mismatch] & high_purity[label_mismatch]
            ]
            if unstable_prompt_candidates.size:
                raise RuntimeError(
                    f"high-purity prompt candidate labels differ for {patch_id}: "
                    f"{json.dumps(label_mismatch_detail)}"
                )
            clean = high_purity & (recomputed == labels)
            positive_clean = clean & (labels == class_id)
            negative_clean = clean & (labels != class_id) & (labels != ignore)
            if not positive_clean.any() or not negative_clean.any():
                continue

            box = (
                int(patch["x_level0"]),
                int(patch["y_level0"]),
                int(patch["width_level0"]),
                int(patch["height_level0"]),
            )
            episode = sample_region_prompt_episode(
                labels,
                active,
                z["fine_centroid_x"],
                z["fine_centroid_y"],
                box,
                rng,
                target_class=class_id,
                min_positive=1,
                max_positive=args.max_positive,
                min_negative=1,
                max_negative=args.max_negative,
                ignore_index=ignore,
                eligible_positive=positive_clean,
                eligible_negative=negative_clean,
                hard_negative_fraction=0.5,
            )
            clicks = []
            for sign, slots in (("positive", episode.positive_slots), ("negative", episode.negative_slots)):
                for slot in slots:
                    prompt_class = int(labels[slot])
                    x, y = interior_point(hard, mask, int(slot), prompt_class)
                    clicks.append(
                        {
                            "sign": sign,
                            "slot": int(slot),
                            "class_id": prompt_class,
                            "class_name": class_names[prompt_class],
                            "x": x,
                            "y": y,
                            "x_normalized": x / mask.shape[1],
                            "y_normalized": y / mask.shape[0],
                            "region_purity": float(purity[slot]),
                            "valid_region_pixels": int(region_pixels[slot]),
                        }
                    )
            for click in clicks:
                if int(hard[click["y"], click["x"]]) != click["slot"]:
                    raise RuntimeError("click-to-slot mapping invariant failed")
                if int(mask[click["y"], click["x"]]) != click["class_id"]:
                    raise RuntimeError("click-to-GT-class invariant failed")

            online_geom = assignment_centroids_areas(assignment, *box)
            both_active = active & online_geom["active"]
            centroid_error = np.hypot(
                z["fine_centroid_x"][both_active] - online_geom["centroid_x"][both_active],
                z["fine_centroid_y"][both_active] - online_geom["centroid_y"][both_active],
            )
            max_centroid_error = float(centroid_error.max(initial=0.0))
            selected = {
                "patch_id": patch_id,
                "wsi_id": patch["wsi_id"],
                "split": args.split,
                "target_class": class_id,
                "target_name": class_names[class_id],
                "candidate_attempt": attempt,
                "positive_slots": episode.positive_slots.astype(int).tolist(),
                "negative_slots": episode.negative_slots.astype(int).tolist(),
                "clicks": clicks,
                "min_required_purity": args.min_purity,
                "max_cached_online_centroid_error_level0": max_centroid_error,
                "region_label_exact_match": not bool(label_mismatch.size),
                "label_mismatch_slots": label_mismatch_detail,
                "prompt_eligible_region_labels_stable": True,
                "click_to_slot_exact_match": True,
                "click_to_gt_class_exact_match": True,
            }
            panel_path = panels / f"class_{class_id:02d}_{class_names[class_id]}__{patch_id}.png"
            quicklook_path = quicklooks / f"class_{class_id:02d}_{class_names[class_id]}__{patch_id}.png"
            save_panel(panel_path, image, mask, hard, labels, class_id, class_names[class_id], clicks, palette, ignore)
            save_quicklook(quicklook_path, image, mask, class_id, clicks)
            selected["panel_path"] = str(panel_path)
            selected["quicklook_path"] = str(quicklook_path)
            (panels / f"class_{class_id:02d}_{class_names[class_id]}__{patch_id}.json").write_text(
                json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            append_jsonl(completed, selected)
            records.append(selected)
            break
        if selected is None:
            raise RuntimeError(
                f"no class {class_id} candidate passed purity={args.min_purity} "
                f"within {len(eligible_patch_ids)} candidates"
            )

    manifest = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"clicks", "positive_slots", "negative_slots"}
            }
            for record in records
        ]
    )
    manifest.to_parquet(output / "prompt_visualization_index.parquet", index=False)
    make_contact_sheet(records, output / "prompt_contact_sheet.png")
    metadata = {
        "timestamp": stamp,
        "purpose": "pre-training human audit only",
        "split": args.split,
        "test_used": False,
        "class_ids": class_ids,
        "class_names": class_names,
        "episode_count": len(records),
        "seed": args.seed,
        "prompt_contract": {
            "task": "one target class versus rest",
            "prompt_unit": "fine-region token selected by a raw point",
            "positive_count": f"1-{args.max_positive}",
            "negative_count": f"1-{args.max_negative} for audit; training contract supports 0-N",
            "clean_region_purity": args.min_purity,
            "ignore_index": ignore,
            "model_input_coordinate": "cached fine-region centroid normalised to the 10x patch",
            "raw_click_coordinate": "interior GT pixel used to map the click to a fine-region slot",
        },
        "inputs": {
            "phase2_config": str(args.phase2_config),
            "phase2_checkpoint": str(args.phase2_checkpoint),
            "patch_index": str(args.patch_index),
            "cache_index": str(args.cache_index),
            "label_index": str(args.label_index),
        },
        "validation": {
            "cached_region_labels_exactly_recomputed": all(r["region_label_exact_match"] for r in records),
            "nonprompt_low_purity_label_mismatch_count": sum(
                len(r["label_mismatch_slots"]) for r in records
            ),
            "all_prompt_eligible_region_labels_stable": all(
                r["prompt_eligible_region_labels_stable"] for r in records
            ),
            "all_clicks_map_to_expected_slot": all(r["click_to_slot_exact_match"] for r in records),
            "all_clicks_lie_on_expected_gt_class": all(r["click_to_gt_class_exact_match"] for r in records),
            "max_cached_online_centroid_error_level0": max(
                r["max_cached_online_centroid_error_level0"] for r in records
            ),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "prompt_visualization_complete", "output": str(output), **metadata["validation"]}))


if __name__ == "__main__":
    main()
