#!/usr/bin/env python3
"""Independent validation audit and six-panel visualization for joint Phase 6."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_3_cell_region.train_phase3 import region_labels
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import IMAGENET_MEAN, IMAGENET_STD
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import EpisodeBalancedSampler, PromptEpisodeDataset
from benchmarks.v4.baseline.common import parse_json_array, validate_episode_manifest
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import (
    binary_counts,
    boundary_f1,
    dice_from_counts,
    summarize_episode_rows,
)
from benchmarks.v4.phase_6_mask_decoder.src.conflict_policy import (
    STRESS_COLUMNS,
    conflict_stress_rows,
)
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import (
    JointPixelEpisodeDataset,
    collate_joint_pixel_episodes,
)
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import JointPromptMaskModel, load_joint_components


HIGH_PURITY_MIN_VALID_PIXELS = 1024
PARENT_CONTEXT_STATE_KEYS = frozenset({
    "parent_context.gate",
    "parent_context.fuse.0.weight",
    "parent_context.fuse.0.bias",
    "parent_context.fuse.1.weight",
    "parent_context.fuse.1.bias",
    "parent_context.fuse.3.weight",
    "parent_context.fuse.3.bias",
})


def cache_mismatch_is_fatal(
    baseline_joint_checkpoint: Path | None, high_purity_mismatch_slots: int
) -> bool:
    """Only the upstream cache reference must reproduce cached region labels.

    An explicit joint baseline can intentionally contain updated geometry.  Its
    cache differences remain audit diagnostics, but are not evidence that the
    checkpoint-to-checkpoint Dice comparison is invalid.
    """
    return baseline_joint_checkpoint is None and high_purity_mismatch_slots > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-joint-checkpoint", type=Path)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--label-index", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cell-routing", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=4000)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--prompt-geometry", type=Path)
    parser.add_argument("--episode-manifest", type=Path,
                        help="Frozen occurrence manifest; overrides sampled prompt count, coordinates, and slots.")
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--pixel-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--threshold-micro-floor", type=float, default=0.7987)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase6/evaluation"))
    parser.add_argument("--timestamp")
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument(
        "--save-all-predictions",
        action="store_true",
        help="Retain prediction arrays for every selected episode, not only chosen panels.",
    )
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


class IndexedEpisodes(Dataset):
    def __init__(self, base: JointPixelEpisodeDataset, indices: list[int]):
        self.base = base; self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        item = self.base[index]; item["episode_index"] = torch.tensor(index, dtype=torch.long)
        return item


class FrozenGeometryEpisodes(Dataset):
    """Apply audited frozen point geometry to one or more base episodes."""
    def __init__(self, base: Dataset, geometry: pd.DataFrame):
        self.base = base; self.geometry = geometry.set_index("episode_index")
    def __len__(self) -> int: return len(self.base)
    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        if int(index) not in self.geometry.index: return item
        row = self.geometry.loc[int(index)]
        positive = np.asarray(json.loads(row.positive_points_10x), np.float32)
        negative = np.asarray(json.loads(row.negative_points_10x), np.float32)
        if len(positive) != int(item["positive_mask"].sum()) or len(negative) != int(item["negative_mask"].sum()):
            raise ValueError(f"frozen prompt count mismatch at episode_index={index}")
        height, width = item["pixel_gt"].shape
        item["positive_xy"][item["positive_mask"]] = torch.from_numpy(positive / np.asarray([width, height], np.float32))
        item["negative_xy"][item["negative_mask"]] = torch.from_numpy(negative / np.asarray([width, height], np.float32))
        return item


class ManifestEpisodes(Dataset):
    """Replay each frozen occurrence with its exact point count and slot identity."""
    def __init__(self, base: JointPixelEpisodeDataset, manifest: pd.DataFrame):
        self.base = base
        self.manifest = manifest.sort_values("occurrence_order").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, position: int) -> dict:
        row = self.manifest.iloc[int(position)]
        episode_index = int(row.episode_index)
        item = self.base[episode_index]
        positive = parse_json_array(row.positive_points_10x, 2)
        negative = parse_json_array(row.negative_points_10x, 2)
        positive_slots = np.asarray(json.loads(row.source_region_ids), dtype=np.int64)
        negative_slots = np.asarray(json.loads(row.negative_source_region_ids), dtype=np.int64)
        if len(positive) != len(positive_slots) or len(negative) != len(negative_slots):
            raise ValueError(f"manifest point/slot mismatch for occurrence {row.occurrence_id}")
        if len(positive) > len(item["positive_mask"]) or len(negative) > len(item["negative_mask"]):
            raise ValueError(f"manifest prompt count exceeds model capacity for occurrence {row.occurrence_id}")
        height, width = item["pixel_gt"].shape
        scale = np.asarray([width, height], np.float32)
        item["positive_xy"].zero_(); item["negative_xy"].zero_()
        item["positive_mask"].fill_(False); item["negative_mask"].fill_(False)
        item["positive_slot_indices"].fill_(-1); item["negative_slot_indices"].fill_(-1)
        item["positive_xy"][:len(positive)] = torch.from_numpy(positive / scale)
        item["negative_xy"][:len(negative)] = torch.from_numpy(negative / scale)
        item["positive_mask"][:len(positive)] = True; item["negative_mask"][:len(negative)] = True
        item["positive_slot_indices"][:len(positive)] = torch.from_numpy(positive_slots)
        item["negative_slot_indices"][:len(negative)] = torch.from_numpy(negative_slots)
        item["prompt_size"] = "point"; item["prompt_size_id"] = torch.tensor(0, dtype=torch.long)
        item["target_class"] = torch.tensor(int(row.target_class), dtype=torch.long)
        item["episode_index"] = torch.tensor(episode_index, dtype=torch.long)
        return item


def construct_model(
    joint_cfg: dict,
    p2cfg: dict,
    p5cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Path | None,
    allow_missing_parent_context: bool = False,
) -> tuple[JointPromptMaskModel, dict]:
    phase2, cell, prompt, transfer = load_joint_components(
        p2cfg, args.phase2_checkpoint, args.cell_checkpoint, args.phase5_checkpoint,
        p5cfg, int(joint_cfg["data"]["cell_feature_dim"]),
    )
    mc = joint_cfg["model"]
    model = JointPromptMaskModel(
        phase2, cell, prompt, int(mc["region_dim"]), int(mc["graph_heads"]),
        int(mc["graph_layers"]), int(mc["graph_neighbours"]), float(mc["graph_dropout"]),
        float(mc["residual_limit"]),
        train_phase2_embedding=bool(mc.get("train_phase2_embedding", False)),
        train_phase2_assignment=bool(mc.get("train_phase2_assignment", False)),
        train_cell=bool(mc.get("train_cell", False)),
        train_prompt=bool(mc.get("train_prompt", False)),
        train_decoder=bool(mc.get("train_decoder", True)),
        train_parent_context=bool(mc.get("train_parent_context", False)),
        train_backbone_layer4=bool(mc.get("train_backbone_layer4", False)),
        num_classes=len(p2cfg["data"]["class_map"]),
        ignore_index=int(joint_cfg["data"]["ignore_index"]),
    )
    load_report = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        missing_set = set(missing); unexpected_set = set(unexpected)
        allowed_missing = PARENT_CONTEXT_STATE_KEYS if allow_missing_parent_context else frozenset()
        if not missing_set.issubset(allowed_missing) or unexpected_set:
            raise RuntimeError(
                f"joint checkpoint load mismatch missing={sorted(missing_set)} "
                f"unexpected={sorted(unexpected_set)} allowed_missing={sorted(allowed_missing)}"
            )
        if missing_set:
            if model.parent_context is None or float(model.parent_context.gate.detach()) != 0.0:
                raise RuntimeError("missing parent-context compatibility requires a zero-gated adapter")
        load_report = {
            "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
            "missing": sorted(missing_set), "unexpected": sorted(unexpected_set),
            "missing_parent_context_zero_equivalent": bool(missing_set),
        }
    return model.to(device).eval(), {"upstream": transfer, "joint": load_report}


def append_counts(row: dict, prefix: str, counts: dict[str, torch.Tensor], position: int) -> None:
    for name in ("tp", "fp", "fn", "tn", "positive", "valid"):
        row[f"{prefix}_{name}"] = int(counts[name][position].item())
    row[f"{prefix}_dice"] = float(
        dice_from_counts(row[f"{prefix}_tp"], row[f"{prefix}_fp"], row[f"{prefix}_fn"])
    )


def compute_metrics(output: dict, batch: dict, ignore: int, tolerance: int) -> dict:
    region_target = output["online_binary_target"]
    region_truth = region_target == 1
    region_valid = region_target != ignore
    unprompted_valid = region_valid & ~output["online_all_prompted_regions"]
    region_prediction = output["logits"] >= 0
    pixel_truth = batch["pixel_gt"] == batch["target_class"][:, None, None]
    pixel_valid = batch["pixel_gt"] != ignore
    pixel_prediction = output["pixel_probability"] >= 0.5
    return {
        "region": binary_counts(region_prediction, region_truth, region_valid),
        "unprompted_region": binary_counts(region_prediction, region_truth, unprompted_valid),
        "pixel": binary_counts(pixel_prediction, pixel_truth, pixel_valid),
        "boundary": boundary_f1(pixel_prediction, pixel_truth, pixel_valid, tolerance),
    }


def pixel_counts_at_threshold(
    output: dict, batch: dict, ignore: int, threshold: float
) -> dict[str, torch.Tensor]:
    truth = batch["pixel_gt"] == batch["target_class"][:, None, None]
    valid = batch["pixel_gt"] != ignore
    return binary_counts(output["pixel_probability"] >= threshold, truth, valid)


def threshold_tag(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "p")


def summarize_threshold_grid(
    frame: pd.DataFrame, thresholds: tuple[float, ...], micro_floor: float
) -> dict:
    result = {"micro_floor": float(micro_floor), "models": {}}
    for model in ("baseline", "joint"):
        model_rows = {}
        for threshold in thresholds:
            prefix = f"{model}_pixel_threshold_{threshold_tag(threshold)}"
            tp = frame[f"{prefix}_tp"].to_numpy(dtype=np.int64)
            fp = frame[f"{prefix}_fp"].to_numpy(dtype=np.int64)
            fn = frame[f"{prefix}_fn"].to_numpy(dtype=np.int64)
            episode_dice = frame[f"{prefix}_dice"].to_numpy(dtype=np.float64)
            micro = float(dice_from_counts(tp.sum(), fp.sum(), fn.sum()).item())
            model_rows[f"{threshold:.3f}"] = {
                "threshold": float(threshold),
                "micro_dice": micro,
                "macro_dice": float(np.nanmean(episode_dice)),
                "evaluable_episodes": int(np.isfinite(episode_dice).sum()),
                "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
                "by_target_class": {
                    str(value): float(part[f"{prefix}_dice"].mean(skipna=True))
                    for value, part in frame.groupby("target_class", sort=True)
                },
                "by_prompt_size": {
                    str(value): float(part[f"{prefix}_dice"].mean(skipna=True))
                    for value, part in frame.groupby("prompt_size", sort=True)
                },
            }
        eligible = [row for row in model_rows.values() if row["micro_dice"] >= micro_floor]
        selected = max(eligible, key=lambda row: (row["macro_dice"], row["micro_dice"])) if eligible else None
        result["models"][model] = {
            "thresholds": model_rows,
            "selected_by_macro": selected,
        }
    return result


def mismatch_quality(
    output: dict, batch: dict, mismatch: torch.Tensor, ignore: int, classes: int
) -> tuple[int, int, float]:
    """Count high-purity/prompted numerical flips and return maximum purity."""
    hard = output["assignment"].argmax(1)
    high_purity = 0; prompted = 0; maximum = 0.0
    for batch_index, slot in mismatch.nonzero(as_tuple=False).tolist():
        pixels = (hard[batch_index] == slot) & (batch["pixel_gt"][batch_index] != ignore)
        if pixels.any():
            votes = torch.bincount(batch["pixel_gt"][batch_index][pixels], minlength=classes)
            purity = float(votes.max().float() / votes.sum().clamp_min(1))
            maximum = max(maximum, purity)
            high_purity += int(purity >= 0.90 and int(pixels.sum()) >= HIGH_PURITY_MIN_VALID_PIXELS)
        prompted += int(batch["all_prompted_regions"][batch_index, slot])
    return high_purity, prompted, maximum


def describe_mismatches(
    output: dict,
    batch: dict,
    online_target: torch.Tensor,
    mismatch: torch.Tensor,
    ignore: int,
    classes: int,
    rank: int,
) -> list[dict]:
    """Create persistent, episode-level evidence for every cache/online flip."""
    hard = output["assignment"].argmax(1)
    records = []
    for batch_index, slot in mismatch.nonzero(as_tuple=False).tolist():
        pixels = (hard[batch_index] == slot) & (batch["pixel_gt"][batch_index] != ignore)
        votes = torch.bincount(
            batch["pixel_gt"][batch_index][pixels], minlength=classes
        ) if pixels.any() else torch.zeros(classes, dtype=torch.long, device=hard.device)
        majority_votes, majority_class = votes.max(0)
        total = int(votes.sum())
        records.append({
            "rank": int(rank),
            "episode_index": int(batch["episode_index"][batch_index]),
            "patch_id": str(batch["patch_id"][batch_index]),
            "target_class": int(batch["target_class"][batch_index]),
            "prompt_size": str(batch["prompt_size"][batch_index]),
            "slot": int(slot),
            "cached_binary_target": int(batch["binary_target"][batch_index, slot]),
            "online_binary_target": int(online_target[batch_index, slot]),
            "majority_class": int(majority_class),
            "majority_votes": int(majority_votes),
            "valid_pixels": total,
            "purity": float(majority_votes.float() / max(total, 1)),
            "positive_prompted": bool(batch["prompted_regions"][batch_index, slot]),
            "negative_prompted": bool(batch["negative_prompted_regions"][batch_index, slot]),
            "online_positive_prompted": bool(output["online_prompted_regions"][batch_index, slot]),
            "online_negative_prompted": bool(output["online_negative_prompted_regions"][batch_index, slot]),
        })
    return records


MISMATCH_COLUMNS = (
    "rank", "episode_index", "patch_id", "target_class", "prompt_size", "slot",
    "cached_binary_target", "online_binary_target", "majority_class", "majority_votes",
    "valid_pixels", "purity", "positive_prompted", "negative_prompted",
    "online_positive_prompted", "online_negative_prompted",
)


def grouped_macro(frame: pd.DataFrame) -> dict:
    result = {}
    for column in ("target_class", "prompt_size"):
        result[f"by_{column}"] = {
            str(value): {
                model: {
                    scope: float(part[f"{model}_{scope}_dice"].mean(skipna=True))
                    for scope in ("region", "unprompted_region", "pixel")
                } | {"boundary_f1": float(part[f"{model}_boundary_f1"].mean(skipna=True))}
                for model in ("baseline", "joint")
            }
            for value, part in frame.groupby(column, sort=True)
        }
    return result


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD[None, None] + IMAGENET_MEAN[None, None]
    return np.clip(image, 0, 1)


def error_rgb(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.full((*truth.shape, 3), 0.08, dtype=np.float32)
    rgb[~valid] = (0.35, 0.35, 0.35)
    rgb[prediction & truth & valid] = (0.15, 0.85, 0.20)
    rgb[prediction & ~truth & valid] = (0.95, 0.15, 0.10)
    rgb[~prediction & truth & valid] = (0.10, 0.35, 1.00)
    return rgb


def add_prompt_markers(ax, batch: dict, width: int, height: int) -> None:
    for sign, xy_key, mask_key, slot_key in (
        ("+", "positive_xy", "positive_mask", "positive_slot_indices"),
        ("-", "negative_xy", "negative_mask", "negative_slot_indices"),
    ):
        xy = batch[xy_key][0].detach().cpu().numpy(); valid = batch[mask_key][0].detach().cpu().numpy()
        slots = batch[slot_key][0].detach().cpu().numpy()
        for (x, y), slot in zip(xy[valid], slots[valid], strict=True):
            px, py = float(x * width), float(y * height)
            positive = sign == "+"
            ax.scatter(px, py, s=80, marker="o" if positive else "X", facecolors="none" if positive else "red",
                       edgecolors="lime" if positive else "white", linewidths=2.0, zorder=6)
            ax.text(px + 5, py - 5, f"{sign}s{int(slot):02d}", color="lime" if positive else "red",
                    fontsize=7, zorder=7)


def render_panel(
    output_path: Path,
    batch: dict,
    baseline: dict,
    joint: dict,
    metrics: dict,
    class_name: str,
) -> None:
    image = denormalize_image(batch["image"][0]); height, width = image.shape[:2]
    gt = batch["pixel_gt"][0].detach().cpu().numpy(); target = int(batch["target_class"][0])
    ignore = 255; truth = gt == target; valid = gt != ignore
    base_prob = baseline["pixel_probability"][0].detach().float().cpu().numpy()
    joint_prob = joint["pixel_probability"][0].detach().float().cpu().numpy()
    base_pred = base_prob >= 0.5; joint_pred = joint_prob >= 0.5
    base_hard = baseline["assignment"][0].argmax(0).detach().cpu().numpy()
    positive = batch["positive_slot_indices"][0][batch["positive_mask"][0]].detach().cpu().numpy()
    negative = batch["negative_slot_indices"][0][batch["negative_mask"][0]].detach().cpu().numpy()
    prompt_overlay = image.copy()
    positive_pixels = np.isin(base_hard, positive); negative_pixels = np.isin(base_hard, negative)
    prompt_overlay[positive_pixels] = 0.55 * prompt_overlay[positive_pixels] + 0.45 * np.array([0.0, 1.0, 0.0])
    prompt_overlay[negative_pixels] = 0.55 * prompt_overlay[negative_pixels] + 0.45 * np.array([1.0, 0.0, 0.0])
    truth_rgb = np.zeros((*truth.shape, 3), dtype=np.float32); truth_rgb[truth] = (1.0, 0.78, 0.05); truth_rgb[~valid] = 0.35

    fig, axes = plt.subplots(1, 6, figsize=(24, 4), constrained_layout=True)
    axes[0].imshow(prompt_overlay); axes[0].set_title("HE + baseline prompt regions")
    axes[1].imshow(truth_rgb); axes[1].set_title(f"Binary GT: {class_name} ({target})")
    for ax in axes[:2]:
        add_prompt_markers(ax, batch, width, height)
    axes[2].imshow(base_prob, cmap="magma", vmin=0, vmax=1); axes[2].contour(truth, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[2].set_title(f"Phase5 probability\nDice={metrics['baseline_pixel_dice']:.3f}")
    axes[3].imshow(joint_prob, cmap="magma", vmin=0, vmax=1); axes[3].contour(truth, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[3].set_title(f"Joint probability\nDice={metrics['joint_pixel_dice']:.3f}")
    axes[4].imshow(error_rgb(base_pred, truth, valid)); axes[4].set_title(
        f"Phase5 errors\nB-F1={metrics['baseline_boundary_f1']:.3f}"
    )
    axes[5].imshow(error_rgb(joint_pred, truth, valid)); axes[5].set_title(
        f"Joint errors\nB-F1={metrics['joint_boundary_f1']:.3f}"
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{batch['patch_id'][0]} | class={class_name} | size={batch['prompt_size'][0]} | "
        "errors: TP green, FP red, FN blue",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    resized = [image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS) if image.width != width else image for image in images]
    canvas = Image.new("RGB", (width, sum(image.height for image in resized)), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y)); y += image.height
    canvas.save(output_path)


def choose_panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    representatives = []
    for (_, _), part in frame.groupby(["target_class", "prompt_size"], sort=True):
        median = float(part["joint_pixel_dice"].median())
        representatives.append(part.loc[(part["joint_pixel_dice"] - median).abs().idxmin()])
    hard = [part.nsmallest(1, "joint_pixel_dice").iloc[0] for _, part in frame.groupby("target_class", sort=True)]
    return pd.DataFrame(representatives), pd.DataFrame(hard)


def source_hashes() -> dict[str, str]:
    files = [
        Path(__file__), Path("benchmarks/v4/phase_6_mask_decoder/src/joint_dataset.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/joint_model.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/model.py"),
        Path("benchmarks/v4/phase_6_mask_decoder/src/conflict_policy.py"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def main() -> None:
    args = parse_args(); cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    thresholds = tuple(dict.fromkeys(float(value) for value in args.pixel_thresholds))
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("pixel thresholds must be unique values strictly between 0 and 1")
    if not 0.0 <= args.threshold_micro_floor <= 1.0:
        raise ValueError("threshold micro floor must be between 0 and 1")
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("joint pixel evaluation requires CUDA")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / (
        f"joint_pixel_audit_test_{stamp}" if args.split == "test" else f"joint_pixel_audit_{stamp}"
    )
    exists = torch.tensor(int(output.exists()), device=device)
    if world > 1:
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if bool(exists.item()):
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True); (output / "shards").mkdir(); (output / "panels").mkdir()
    if world > 1:
        dist.barrier()

    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, args.split, int(cfg["project"]["seed"]),
        cfg["data"]["size_probabilities"], tuple(cfg["data"]["class_ids"]), int(cfg["data"]["ignore_index"]),
        int(cfg["data"]["centroid_knn"]), args.eligibility_index,
    )
    sampler = EpisodeBalancedSampler(
        episodes, cfg["data"]["size_probabilities"], cfg["data"]["group_probabilities"],
        tuple(cfg["data"]["class_ids"]), int(cfg["project"]["seed"]) + 10_000_019,
        epoch_size=min(int(args.validation_episodes), len(episodes)),
    )
    selected = list(iter(sampler))
    if args.episode_index is not None:
        if args.episode_index < 0 or args.episode_index >= len(episodes): raise IndexError(args.episode_index)
        selected = [int(args.episode_index)]
    local_indices = selected[rank::world]
    full = JointPixelEpisodeDataset(
        episodes, args.patch_index, args.cell_routing, p2cfg["data"]["class_map"],
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]),
        include_parent_context=bool(cfg["model"].get("train_parent_context", False)),
    )
    if args.prompt_geometry is not None and args.episode_manifest is not None:
        raise ValueError("--prompt-geometry and --episode-manifest are mutually exclusive")
    if args.prompt_geometry is not None:
        full = FrozenGeometryEpisodes(full, pd.read_parquet(args.prompt_geometry))
    if args.episode_manifest is not None:
        manifest = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
        if len(manifest) > int(args.validation_episodes):
            manifest = manifest.iloc[:int(args.validation_episodes)].copy()
            manifest["occurrence_order"] = np.arange(len(manifest), dtype=np.int64)
        audit = validate_episode_manifest(manifest, args.split)
        full = ManifestEpisodes(full, manifest)
        selected = list(range(len(full)))
        local_indices = selected[rank::world]
        local = IndexedEpisodes(full, local_indices)
    else:
        local = IndexedEpisodes(full, local_indices)
    collate = functools.partial(collate_joint_pixel_episodes, max_cells=int(cfg["data"]["max_cells_per_patch"]))
    loader_args = dict(batch_size=int(args.batch_size_per_gpu), num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate)
    if args.num_workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(local, shuffle=False, **loader_args)

    baseline, baseline_load = construct_model(
        cfg, p2cfg, p5cfg, args, device, args.baseline_joint_checkpoint,
        allow_missing_parent_context=args.baseline_joint_checkpoint is not None,
    )
    joint, joint_load = construct_model(cfg, p2cfg, p5cfg, args, device, args.joint_checkpoint)
    ignore = int(cfg["data"]["ignore_index"]); classes = len(p2cfg["data"]["class_map"])
    rows = []; baseline_mismatch = 0; baseline_high_purity = 0; baseline_prompted = 0
    baseline_max_mismatch_purity = 0.0; joint_changed = 0; compared = 0; t0 = time.monotonic()
    baseline_positive_changes = 0; baseline_negative_changes = 0; mismatch_rows = []
    baseline_prompt_conflicts = 0; joint_prompt_conflicts = 0
    baseline_conflict_episodes = 0; joint_conflict_episodes = 0; stress_rows = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                baseline_output = baseline(batch); joint_output = joint(batch)
            baseline_metrics = compute_metrics(baseline_output, batch, ignore, int(args.boundary_tolerance))
            joint_metrics = compute_metrics(joint_output, batch, ignore, int(args.boundary_tolerance))
            threshold_metrics = {
                (name, threshold): pixel_counts_at_threshold(output_item, batch, ignore, threshold)
                for name, output_item in (("baseline", baseline_output), ("joint", joint_output))
                for threshold in thresholds
            }
            for output_item, accumulator in ((baseline_output, "baseline"), (joint_output, "joint")):
                labels = region_labels(output_item["assignment_low"], batch["pixel_gt"], classes, ignore)
                target = torch.full_like(labels, ignore); active = batch["fine_active"] & (labels != ignore)
                target[active] = (labels[active] == batch["target_class"][:, None].expand_as(labels)[active]).long()
                mismatch_mask = (target != batch["binary_target"]) & batch["fine_active"]
                mismatch_count = int(mismatch_mask.sum())
                if accumulator == "baseline":
                    baseline_mismatch += mismatch_count
                    high_purity, prompted, maximum = mismatch_quality(
                        output_item, batch, mismatch_mask, ignore, classes
                    )
                    baseline_high_purity += high_purity; baseline_prompted += prompted
                    baseline_max_mismatch_purity = max(baseline_max_mismatch_purity, maximum)
                    mismatch_rows.extend(describe_mismatches(
                        output_item, batch, target, mismatch_mask, ignore, classes, rank
                    ))
                else:
                    joint_changed += mismatch_count
            baseline_positive_changes += int((
                (baseline_output["online_positive_slot_indices"] != batch["positive_slot_indices"])
                & batch["positive_mask"]
            ).sum())
            baseline_negative_changes += int((
                (baseline_output["online_negative_slot_indices"] != batch["negative_slot_indices"])
                & batch["negative_mask"]
            ).sum())
            baseline_prompt_conflicts += int(baseline_output["online_prompt_conflicts"].sum())
            joint_prompt_conflicts += int(joint_output["online_prompt_conflicts"].sum())
            baseline_conflict_episodes += int(baseline_output["online_prompt_conflicts"].any(1).sum())
            joint_conflict_episodes += int(joint_output["online_prompt_conflicts"].any(1).sum())
            stress_rows.extend(conflict_stress_rows(joint_output, batch, epoch=-1, rank=rank))
            compared += int(batch["fine_active"].sum())
            for position, episode_index in enumerate(batch["episode_index"]):
                row = {
                    "episode_index": int(episode_index), "patch_id": batch["patch_id"][position],
                    "wsi_id": batch["wsi_id"][position], "target_class": int(batch["target_class"][position]),
                    "prompt_size": batch["prompt_size"][position], "rank": rank,
                }
                for name, computed in (("baseline", baseline_metrics), ("joint", joint_metrics)):
                    for scope in ("region", "unprompted_region", "pixel"):
                        append_counts(row, f"{name}_{scope}", computed[scope], position)
                    row[f"{name}_boundary_f1"] = float(computed["boundary"]["boundary_f1"][position])
                    row[f"{name}_boundary_evaluable"] = bool(computed["boundary"]["boundary_evaluable"][position])
                    for threshold in thresholds:
                        append_counts(
                            row, f"{name}_pixel_threshold_{threshold_tag(threshold)}",
                            threshold_metrics[(name, threshold)], position,
                        )
                rows.append(row)
            if step % 100 == 0:
                print({"event": "audit_progress", "rank": rank, "batches": step, "episodes": min(step * args.batch_size_per_gpu, len(local))}, flush=True)
    pd.DataFrame(rows).to_parquet(output / "shards" / f"episodes_rank{rank:02d}.parquet", index=False)
    pd.DataFrame(mismatch_rows, columns=MISMATCH_COLUMNS).to_parquet(
        output / "shards" / f"mismatches_rank{rank:02d}.parquet", index=False
    )
    pd.DataFrame(stress_rows, columns=STRESS_COLUMNS).to_parquet(
        output / "shards" / f"stress_rank{rank:02d}.parquet", index=False
    )
    local_audit = {
        "counts": [baseline_mismatch, baseline_high_purity, baseline_prompted, joint_changed, compared,
                   baseline_positive_changes, baseline_negative_changes, baseline_prompt_conflicts,
                   joint_prompt_conflicts, baseline_conflict_episodes, joint_conflict_episodes],
        "baseline_max_mismatch_purity": baseline_max_mismatch_purity,
    }
    save_json(output / "shards" / f"audit_rank{rank:02d}.json", local_audit)
    if rank == 0:
        deadline = time.monotonic() + 300.0
        while len(list((output / "shards").glob("audit_rank*.json"))) < world:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {world} audit shards")
            time.sleep(0.2)
        audit_parts = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "shards").glob("audit_rank*.json"))
        ]
        if len(audit_parts) != world:
            raise RuntimeError(f"found {len(audit_parts)} audit shards, expected {world}")
        audit_counts = np.asarray([part["counts"] for part in audit_parts], dtype=np.int64).sum(0)
        audit_maximum = max(float(part["baseline_max_mismatch_purity"]) for part in audit_parts)
        frame = pd.concat([pd.read_parquet(path) for path in sorted((output / "shards").glob("episodes_rank*.parquet"))], ignore_index=True)
        mismatch_frame = pd.concat(
            [pd.read_parquet(path) for path in sorted((output / "shards").glob("mismatches_rank*.parquet"))],
            ignore_index=True,
        )
        if len(frame) != len(selected):
            raise RuntimeError(f"merged {len(frame)} episodes, expected {len(selected)}")
        frame.to_parquet(output / "episode_metrics.parquet", index=False)
        mismatch_frame.to_parquet(output / "mismatch_details.parquet", index=False)
        stress = pd.concat(
            [pd.read_parquet(path) for path in sorted((output / "shards").glob("stress_rank*.parquet"))],
            ignore_index=True,
        )
        if len(stress) != int(audit_counts[10]):
            raise RuntimeError(
                f"joint stress set has {len(stress)} rows, expected {int(audit_counts[10])}"
            )
        stress.to_parquet(output / "stress_set.parquet", index=False)
        if cache_mismatch_is_fatal(args.baseline_joint_checkpoint, int(audit_counts[1])):
            raise RuntimeError(
                f"baseline/cache mismatch touches substantial high-purity={int(audit_counts[1])} "
                f"slots; recorded prompted mismatches={int(audit_counts[2])}; prompt remap changes "
                f"positive={int(audit_counts[5])} negative={int(audit_counts[6])} "
                f"conflicts={int(audit_counts[7])}"
            )
        summary = summarize_episode_rows(frame.to_dict("records"), ("baseline", "joint"))
        summary.update(grouped_macro(frame)); summary["pixel_threshold_grid"] = summarize_threshold_grid(
            frame, thresholds, float(args.threshold_micro_floor)
        ); summary.update({
            "timestamp": stamp, "split": args.split, "test_used": args.split == "test", "world_size": world,
            "baseline_region_label_mismatch_slots": int(audit_counts[0]),
            "baseline_high_purity_mismatch_slots": int(audit_counts[1]),
            "baseline_prompted_mismatch_slots": int(audit_counts[2]),
            "baseline_max_mismatch_purity": float(audit_maximum),
            "high_purity_mismatch_min_valid_pixels": HIGH_PURITY_MIN_VALID_PIXELS,
            "baseline_is_cache_reference": args.baseline_joint_checkpoint is None,
            "joint_region_label_changed_slots": int(audit_counts[3]), "compared_slots": int(audit_counts[4]),
            "baseline_positive_prompt_slot_changes": int(audit_counts[5]),
            "baseline_negative_prompt_slot_changes": int(audit_counts[6]),
            "baseline_prompt_conflict_slots": int(audit_counts[7]),
            "joint_prompt_conflict_slots": int(audit_counts[8]),
            "baseline_prompt_conflict_episodes": int(audit_counts[9]),
            "joint_prompt_conflict_episodes": int(audit_counts[10]),
            "baseline_prompt_conflict_episode_rate": int(audit_counts[9]) / max(len(selected), 1),
            "joint_prompt_conflict_episode_rate": int(audit_counts[10]) / max(len(selected), 1),
            "joint_stress_set_path": str(output / "stress_set.parquet"),
            "panels_skipped": bool(args.skip_panels),
            "save_all_predictions": bool(args.save_all_predictions),
            "pixel_thresholds": list(thresholds),
            "elapsed_seconds": time.monotonic() - t0, "baseline_load": baseline_load, "joint_load": joint_load,
            "inputs": {key: str(getattr(args, key)) for key in (
                "config", "phase2_config", "phase5_config", "phase2_checkpoint", "cell_checkpoint",
                "phase5_checkpoint", "baseline_joint_checkpoint", "joint_checkpoint", "cache_index", "label_index", "patch_index",
                "eligibility_index", "cell_routing", "episode_manifest",
            )},
            "reproducibility": {"command": [sys.executable, *sys.argv], "source_sha256": source_hashes()},
        })
        panel_records = []
        if args.save_all_predictions:
            for episode_index in sorted(frame["episode_index"].astype(int).unique()):
                item = full[episode_index]
                item["episode_index"] = torch.tensor(episode_index)
                batch = move_batch(collate([item]), device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    joint_output = joint(batch)
                np.savez_compressed(
                    output / f"prediction_arrays_{episode_index:06d}.npz",
                    probability=joint_output["pixel_probability"][0].float().cpu().numpy(),
                    binary_mask=(joint_output["pixel_probability"][0] >= 0.5).cpu().numpy(),
                    target_mask=(batch["pixel_gt"][0] == batch["target_class"][0]).cpu().numpy(),
                    image=denormalize_image(batch["image"][0]),
                    positive_points=batch["positive_xy"][0][batch["positive_mask"][0]].cpu().numpy()
                    * [batch["image"].shape[-1], batch["image"].shape[-2]],
                    negative_points=batch["negative_xy"][0][batch["negative_mask"][0]].cpu().numpy()
                    * [batch["image"].shape[-1], batch["image"].shape[-2]],
                )
        if not args.skip_panels:
            representatives, hard = choose_panels(frame)
            class_names = [item.get("name", f"class_{index}") for index, item in enumerate(p2cfg["data"]["class_map"])]
            for group_name, selection in (("representative", representatives), ("hard", hard)):
                for _, row in selection.iterrows():
                    episode_index = int(row["episode_index"]); item = full[episode_index]; item["episode_index"] = torch.tensor(episode_index)
                    batch = move_batch(collate([item]), device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        baseline_output = baseline(batch); joint_output = joint(batch)
                    np.savez_compressed(output / f"prediction_arrays_{episode_index:06d}.npz",
                        probability=joint_output["pixel_probability"][0].float().cpu().numpy(),
                        binary_mask=(joint_output["pixel_probability"][0] >= 0.5).cpu().numpy(),
                        target_mask=(batch["pixel_gt"][0] == batch["target_class"][0]).cpu().numpy(),
                        image=denormalize_image(batch["image"][0]),
                        positive_points=batch["positive_xy"][0][batch["positive_mask"][0]].cpu().numpy() * [batch["image"].shape[-1], batch["image"].shape[-2]],
                        negative_points=batch["negative_xy"][0][batch["negative_mask"][0]].cpu().numpy() * [batch["image"].shape[-1], batch["image"].shape[-2]])
                    filename = f"{group_name}_c{int(row['target_class']):02d}_{row['prompt_size']}_{episode_index:06d}.png"
                    render_panel(output / "panels" / filename, batch, baseline_output, joint_output, row.to_dict(), class_names[int(row["target_class"])])
                    panel_records.append({"group": group_name, "episode_index": episode_index, "path": f"panels/{filename}", **row.to_dict()})
            for size in ("point", "small", "large"):
                paths = [output / record["path"] for record in panel_records if record["group"] == "representative" and record["prompt_size"] == size]
                make_contact_sheet(paths, output / f"contact_sheet_{size}.png")
            make_contact_sheet([output / record["path"] for record in panel_records if record["group"] == "hard"], output / "contact_sheet_hard_cases.png")
        save_json(output / "metadata.json", {"summary": summary, "panels": panel_records})
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
