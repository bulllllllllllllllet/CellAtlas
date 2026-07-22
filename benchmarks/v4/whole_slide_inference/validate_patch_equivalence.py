#!/usr/bin/env python3
"""Validate reusable WSI prompt-task decoding against the original J5 patch path."""
from __future__ import annotations

import argparse
import functools
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import (
    JointPixelEpisodeDataset,
    collate_joint_pixel_episodes,
)
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import remap_prompt_tokens
from benchmarks.v4.whole_slide_inference.infer_wsi import load_model
from benchmarks.v4.whole_slide_inference.src.prompt_transfer import (
    EncodedRegions,
    decode_regions_with_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/whole_slide_inference/validation"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1 or not torch.cuda.is_available():
        raise RuntimeError("validation requires exactly one available CUDA GPU")
    torch.cuda.set_device(args.gpus[0]); device = torch.device("cuda", args.gpus[0])
    cfg = load_config(args.config); p2cfg = load_config(args.phase2_config)
    episodes = PromptEpisodeDataset(
        args.cache_index, args.label_index, args.patch_index, "val", int(cfg["project"]["seed"]),
        cfg["data"]["size_probabilities"], tuple(cfg["data"]["class_ids"]),
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["centroid_knn"]),
        args.eligibility_index,
    )
    dataset = JointPixelEpisodeDataset(
        episodes, args.patch_index, args.cell_routing, p2cfg["data"]["class_map"],
        int(cfg["data"]["ignore_index"]), int(cfg["data"]["max_cells_per_patch"]),
        include_parent_context=False,
    )
    collate = functools.partial(
        collate_joint_pixel_episodes, max_cells=int(cfg["data"]["max_cells_per_patch"])
    )
    batch = move_batch(collate([dataset[int(args.episode_index)]]), device)
    model, load_report = load_model(args, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct = model(batch)
        positive_tokens = remap_prompt_tokens(
            direct["online_contextual_tokens"], direct["online_region_xy"],
            batch["positive_xy"], batch["positive_mask"],
        )[0]
        negative_tokens = remap_prompt_tokens(
            direct["online_contextual_tokens"], direct["online_region_xy"],
            batch["negative_xy"], batch["negative_mask"],
        )[0]
        prompt_batch = {
            "positive_tokens": positive_tokens, "negative_tokens": negative_tokens,
            "positive_xy": batch["positive_xy"], "negative_xy": batch["negative_xy"],
            "positive_mask": batch["positive_mask"], "negative_mask": batch["negative_mask"],
            "prompt_size_id": batch["prompt_size_id"],
        }
        task = model.decoder.prompt_model.encode_prompt_task(prompt_batch)
        active = direct["online_region_area"] > 1e-6
        if not torch.equal(active, batch["fine_active"]):
            raise RuntimeError("online active-region rule differs from the audited cache")
        transferred = decode_regions_with_task(
            model,
            EncodedRegions(
                direct["assignment"], direct["online_contextual_tokens"],
                direct["online_region_xy"], direct["online_region_area"], active,
            ),
            task,
        )
    probability_error = float((direct["pixel_probability"] - transferred["pixel_probability"]).abs().max())
    logit_error = float((direct["logits"] - transferred["logits"]).abs().max())
    if probability_error != 0.0 or logit_error != 0.0:
        raise RuntimeError(f"prompt transfer is not exact: probability={probability_error} logits={logit_error}")
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"patch_equivalence_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {
        "timestamp": stamp, "split": "val", "test_used": False,
        "episode_index": int(args.episode_index), "patch_id": str(batch["patch_id"][0]),
        "prompt_size": str(batch["prompt_size"][0]),
        "probability_max_abs_error": probability_error, "region_logit_max_abs_error": logit_error,
        "active_regions_equal": True, "load_report": load_report, "status": "passed",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
