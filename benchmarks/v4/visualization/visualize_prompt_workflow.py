#!/usr/bin/env python3
"""Render validation-only prompt-to-mask workflow figures for a joint checkpoint."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts, dice_from_counts
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import (
    JointPixelEpisodeDataset,
    collate_joint_pixel_episodes,
)
from benchmarks.v4.phase_6_mask_decoder.tools.evaluate_visualize_joint_pixel import (
    construct_model,
    denormalize_image,
    error_rgb,
    move_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--cell-routing", type=Path, required=True)
    parser.add_argument("--episode-metrics", type=Path, required=True)
    parser.add_argument("--stress-set", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase6/visualization"))
    parser.add_argument("--timestamp")
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def region_edges(hard: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(hard, dtype=bool)
    vertical = hard[1:] != hard[:-1]
    horizontal = hard[:, 1:] != hard[:, :-1]
    edge[1:] |= vertical; edge[:-1] |= vertical
    edge[:, 1:] |= horizontal; edge[:, :-1] |= horizontal
    return edge


def tint_regions(
    image: np.ndarray, hard: np.ndarray, slots: np.ndarray, color: tuple[float, float, float]
) -> np.ndarray:
    result = image.copy(); selected = np.isin(hard, slots)
    result[selected] = 0.48 * result[selected] + 0.52 * np.asarray(color)
    return result


def plot_points(ax, batch: dict, positive: bool) -> list[dict]:
    xy_key = "positive_xy" if positive else "negative_xy"
    mask_key = "positive_mask" if positive else "negative_mask"
    slot_key = "online_positive_slot_indices" if positive else "online_negative_slot_indices"
    xy = batch[xy_key][0][batch[mask_key][0]].detach().float().cpu().numpy()
    slots = batch[slot_key][0][batch[mask_key][0]].detach().cpu().numpy()
    marker = "o" if positive else "X"; color = "lime" if positive else "red"
    records = []
    for order, ((x, y), slot) in enumerate(zip(xy, slots, strict=True), start=1):
        ax.scatter(float(x) * 512, float(y) * 512, marker=marker, s=95,
                   facecolors="none" if positive else color, edgecolors=color,
                   linewidths=2.0, zorder=5)
        ax.text(float(x) * 512 + 5, float(y) * 512 - 5, f"{order}:r{int(slot)}",
                color=color, fontsize=7, weight="bold", zorder=6)
        records.append({"order": order, "x_normalized": float(x), "y_normalized": float(y), "region_slot": int(slot)})
    return records


def select_examples(frame: pd.DataFrame, conflicts: set[int], limit: int) -> list[pd.Series]:
    safe = frame[~frame["episode_index"].isin(conflicts)].drop_duplicates("episode_index")
    selected = []
    for size in ("point", "small", "large"):
        part = safe[safe["prompt_size"] == size]
        if part.empty:
            raise ValueError(f"no non-conflict validation episode for prompt size {size}")
        median = float(part["baseline_pixel_dice"].median())
        selected.append(part.loc[(part["baseline_pixel_dice"] - median).abs().idxmin()])
    return selected[:limit]


def audited_batch_rows(frame: pd.DataFrame, row: pd.Series, batch_size: int = 2) -> tuple[pd.DataFrame, int]:
    """Recover the original per-rank batch used by the DDP paired audit."""
    rank_rows = frame[frame["rank"] == int(row["rank"])]
    positions = np.flatnonzero(rank_rows.index.to_numpy() == int(row.name))
    if len(positions) != 1:
        raise RuntimeError("selected audit occurrence is not unique within its rank")
    position = int(positions[0]); start = (position // int(batch_size)) * int(batch_size)
    batch_rows = rank_rows.iloc[start:start + int(batch_size)]
    if len(batch_rows) != int(batch_size):
        raise RuntimeError("selected audit occurrence belongs to an incomplete replay batch")
    return batch_rows, position - start


def take_batch_position(values: dict, position: int, batch_size: int) -> dict:
    selected = {}
    for key, value in values.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            selected[key] = value[position:position + 1]
        elif isinstance(value, list) and len(value) == batch_size:
            selected[key] = [value[position]]
        else:
            selected[key] = value
    return selected


def render(
    path: Path,
    batch: dict,
    output: dict,
    class_name: str,
    expected_dice: float,
) -> dict:
    image = denormalize_image(batch["image"][0]); gt = batch["pixel_gt"][0].detach().cpu().numpy()
    target = int(batch["target_class"][0]); valid = gt != 255; truth = gt == target
    hard = output["assignment"][0].argmax(0).detach().cpu().numpy()
    probability = output["pixel_probability"][0].detach().float().cpu().numpy()
    prediction = probability >= 0.5
    positive = output["online_prompted_regions"][0].nonzero().flatten().detach().cpu().numpy()
    negative = output["online_negative_prompted_regions"][0].nonzero().flatten().detach().cpu().numpy()
    conflict = bool(output["online_prompt_conflicts"][0].any())
    counts = binary_counts(
        torch.from_numpy(prediction)[None], torch.from_numpy(truth)[None], torch.from_numpy(valid)[None]
    )
    dice = float(dice_from_counts(int(counts["tp"][0]), int(counts["fp"][0]), int(counts["fn"][0])))
    if abs(dice - float(expected_dice)) > 1e-6:
        raise RuntimeError(f"rendered Dice {dice} differs from audited episode metric {expected_dice}")

    region_view = image.copy(); region_view[region_edges(hard)] = (0.0, 1.0, 1.0)
    positive_view = tint_regions(image, hard, positive, (0.0, 1.0, 0.0))
    negative_view = tint_regions(image, hard, negative, (1.0, 0.0, 0.0))
    gt_view = np.zeros((*truth.shape, 3), dtype=np.float32); gt_view[truth] = (1.0, 0.82, 0.0); gt_view[~valid] = 0.35
    mask_view = np.zeros((*prediction.shape, 3), dtype=np.float32); mask_view[prediction] = (0.0, 0.85, 1.0)

    fig, axes = plt.subplots(1, 8, figsize=(28, 3.8), constrained_layout=True)
    axes[0].imshow(image); axes[0].set_title("1. HE patch")
    axes[1].imshow(region_view); axes[1].set_title("2. Learned regions")
    axes[2].imshow(positive_view); axes[2].set_title("3. Positive prompt")
    positive_records = plot_points(axes[2], output | batch, True)
    axes[3].imshow(negative_view); axes[3].set_title("4. Negative prompt")
    negative_records = plot_points(axes[3], output | batch, False)
    axes[4].imshow(probability, cmap="magma", vmin=0, vmax=1); axes[4].set_title("5. Target probability")
    if conflict:
        axes[5].imshow(np.full((*prediction.shape, 3), 0.18)); axes[5].text(
            0.5, 0.5, "ABSTAIN\nadjust prompt", transform=axes[5].transAxes,
            ha="center", va="center", color="white", fontsize=16, weight="bold"
        ); axes[5].set_title("6. No mask returned")
    else:
        axes[5].imshow(mask_view); axes[5].set_title(f"6. Final mask @0.5\nDice={dice:.4f}")
    axes[6].imshow(gt_view); axes[6].set_title(f"7. GT: {class_name}")
    axes[7].imshow(error_rgb(prediction, truth, valid)); axes[7].set_title(
        "8. Error audit\nTP green / FP red / FN blue" + ("\n(raw; output abstained)" if conflict else "")
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{batch['patch_id'][0]} | class={class_name} | prompt={batch['prompt_size'][0]} | "
        f"conflict={conflict}", fontsize=11,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return {
        "path": str(path), "episode_index": int(batch["episode_index"][0]),
        "patch_id": str(batch["patch_id"][0]), "wsi_id": str(batch["wsi_id"][0]),
        "target_class": target, "class_name": class_name, "prompt_size": str(batch["prompt_size"][0]),
        "positive_prompts": positive_records, "negative_prompts": negative_records,
        "online_positive_regions": positive.tolist(), "online_negative_regions": negative.tolist(),
        "prompt_conflict": conflict, "mask_returned": not conflict, "pixel_dice": dice,
    }


def contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    resized = [image if image.width == width else image.resize(
        (width, round(image.height * width / image.width)), Image.Resampling.LANCZOS
    ) for image in images]
    canvas = Image.new("RGB", (width, sum(image.height for image in resized)), "white")
    top = 0
    for image in resized:
        canvas.paste(image, (0, top)); top += image.height
    canvas.save(output)


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"prompt_workflow_{stamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("workflow visualization requires CUDA")
    if not 1 <= int(args.max_examples) <= 3:
        raise ValueError("max-examples must be between 1 and 3")
    cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    frame = pd.read_parquet(args.episode_metrics)
    stress = pd.read_parquet(args.stress_set)
    selected = select_examples(frame, set(map(int, stress["episode_index"])), int(args.max_examples))
    conflict_row = frame[frame["episode_index"].isin(set(map(int, stress["episode_index"])))].iloc[0]
    device = torch.device("cuda", 0); torch.cuda.set_device(0)
    model, load_report = construct_model(
        cfg, p2cfg, p5cfg, args, device, args.joint_checkpoint,
        allow_missing_parent_context=True,
    )
    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, "val", int(cfg["project"]["seed"]),
        cfg["data"]["size_probabilities"], tuple(cfg["data"]["class_ids"]), int(cfg["data"]["ignore_index"]),
        int(cfg["data"]["centroid_knn"]), args.eligibility_index,
    )
    dataset = JointPixelEpisodeDataset(
        episodes, args.patch_index, args.cell_routing, p2cfg["data"]["class_map"],
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]),
        include_parent_context=bool(cfg["model"].get("train_parent_context", False)),
    )
    collate = functools.partial(collate_joint_pixel_episodes, max_cells=int(cfg["data"]["max_cells_per_patch"]))
    names = {int(item["id"]): str(item["name"]) for item in p2cfg["data"]["class_map"]}
    output_dir.mkdir(parents=True)
    records = []; paths = []
    for order, row in enumerate([*selected, conflict_row]):
        replay_rows, target_position = audited_batch_rows(frame, row)
        items = [dataset[int(index)] for index in replay_rows["episode_index"]]
        replay_batch = move_batch(collate(items), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            replay_result = model(replay_batch)
        batch = take_batch_position(replay_batch, target_position, len(items))
        result = take_batch_position(replay_result, target_position, len(items))
        label = "conflict_abstain" if order == len(selected) else str(row["prompt_size"])
        path = output_dir / f"workflow_{order:02d}_{label}.png"
        records.append(render(path, batch, result, names[int(row["target_class"])], float(row["baseline_pixel_dice"])))
        paths.append(path)
    contact_sheet(paths, output_dir / "workflow_contact_sheet.png")
    metadata = {
        "timestamp": stamp, "split": "val", "test_used": False,
        "checkpoint": str(args.joint_checkpoint), "checkpoint_sha256": sha256(args.joint_checkpoint),
        "episode_metrics": str(args.episode_metrics), "episode_metrics_sha256": sha256(args.episode_metrics),
        "stress_set": str(args.stress_set), "load_report": load_report,
        "examples": records, "contact_sheet": str(output_dir / "workflow_contact_sheet.png"),
        "scope": "patch-level 512x512 prompt-conditioned segmentation; no WSI stitching",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
