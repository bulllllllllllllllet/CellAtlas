#!/usr/bin/env python3
"""Print or launch a contract-checked V4 ablation training command."""
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
J5_BEST = Path("/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth")
COMMON = {
    "phase2_config": ROOT / "phase_2_region_encoder/configs/phase2_region_10x.yaml",
    "phase5_config": ROOT / "phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml",
    "phase2_checkpoint": Path("/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth"),
    "cell_checkpoint": Path("/nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth"),
    "phase5_checkpoint": Path("/nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth"),
    "cache_index": Path("/nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet"),
    "label_index": Path("/nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet"),
    "patch_index": Path("/nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet"),
    "eligibility_index": Path("/nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet"),
    "train_cell_routing": Path("/nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_train_20260719_014500_spatial/feature_routing.parquet"),
    "val_cell_routing": Path("/nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet"),
    "slic_index": Path("/nfs-medical3/zyh/v4/phase2/data/slic_mmap_trainval_20260717_151426/slic_index.parquet"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("phase4_attention", "phase4_attention_shuffled", "cell_branch_removed", "phase2_fixed_slic", "prompt_mean_prototype"), required=True)
    parser.add_argument("--gpus", default="0,2,3,4")
    parser.add_argument("--timestamp")
    parser.add_argument("--execute", action="store_true", help="launch after validation; otherwise print only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ROOT / "ablation/configs" / f"{args.experiment}.yaml"
    payload = yaml.safe_load(config.read_text())
    if payload["checkpoint_selection"]["dice_reference"]["pixel_macro_dice"] != 0.7413279258440644:
        raise ValueError("ablation config is not anchored to the frozen J5 validation reference")
    if payload["project"]["name"] != f"ablation_{args.experiment}":
        raise ValueError("project name must match the selected ablation")
    if not J5_BEST.is_file():
        raise FileNotFoundError(J5_BEST)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    command = [
        "conda", "run", "-n", "aligner", "torchrun", "--standalone",
        f"--nproc_per_node={len(args.gpus.split(','))}", "-m", "benchmarks.v4.phase_6_mask_decoder.train_joint_pixel",
        "--config", str(config), "--initial-joint-checkpoint", str(J5_BEST), "--timestamp", stamp,
    ]
    for name, path in COMMON.items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    print("CUDA_VISIBLE_DEVICES=" + args.gpus)
    print(" ".join(command))
    if args.execute:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        subprocess.run(command, cwd=ROOT.parents[1], env=env, check=True)


if __name__ == "__main__":
    main()
