#!/usr/bin/env python3
"""Run the frozen 1+1, 3+1, and 5+1 Phase-6 point-prompt curve sequentially."""
from __future__ import annotations

import os
import subprocess
import sys


ROOT = "/home/zhaoyh/CellAtlas"
COMMON = [
    sys.executable, "-m", "benchmarks.v4.phase_6_mask_decoder.tools.evaluate_visualize_joint_pixel",
    "--config", "benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml",
    "--phase2-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
    "--phase5-config", "benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml",
    "--phase2-checkpoint", "/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth",
    "--cell-checkpoint", "/nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth",
    "--phase5-checkpoint", "/nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth",
    "--joint-checkpoint", "/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth",
    "--baseline-joint-checkpoint", "/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth",
    "--cache-index", "/nfs-medical3/zyh/v4/phase4/data/multiscale_token_cache_20260719_205544/cache_index.parquet",
    "--label-index", "/nfs-medical3/zyh/v4/phase4/data/fine_region_labels_20260719_230128/label_index.parquet",
    "--patch-index", "/nfs-medical3/zyh/v4/phase1/data/multiscale_index_20260716_150251/patch_index_10x_5x_2p5x.parquet",
    "--eligibility-index", "/nfs-medical3/zyh/v4/phase5/data/prompt_eligibility_20260720_210851/eligibility_index.parquet",
    "--cell-routing", "/nfs-medical3/zyh/v4/phase3/data/xcell_hybrid_manifest_val_20260719_014500_spatial/feature_routing.parquet",
    "--split", "val", "--batch-size-per-gpu", "1", "--num-workers", "0", "--skip-panels",
    "--output-root", "/nfs-medical3/zyh/v4/phase6/evaluation",
]
RUNS = [
    ("/nfs-medical3/zyh/v4/baseline/point_1p_1n_manifest_20260724_173500/episode_manifest_20260724_173500.parquet", "4000", "20260724_173800"),
    ("/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet", "1261", "20260724_173801"),
    ("/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet", "885", "20260724_173802"),
]

for manifest, count, stamp in RUNS:
    command = COMMON + ["--episode-manifest", manifest, "--validation-episodes", count, "--timestamp", stamp]
    print({"event": "starting", "timestamp": stamp, "episodes": int(count)}, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
