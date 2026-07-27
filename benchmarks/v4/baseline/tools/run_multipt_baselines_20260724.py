#!/usr/bin/env python3
"""Run frozen 3+1 and 5+1 external-baseline evaluations sequentially."""
from __future__ import annotations

import subprocess
import sys


ROOT = "/home/zhaoyh/CellAtlas"
RUNS = (
    ("sam_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet", "20260724_200000"),
    ("sam_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet", "20260724_200001"),
    ("sam_med2d_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet", "20260724_200010"),
    ("sam_med2d_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet", "20260724_200011"),
    ("wsi_sam_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet", "20260724_200020"),
    ("wsi_sam_zero_shot.yaml", "/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet", "20260724_200021"),
)
for config, manifest, stamp in RUNS:
    command = [
        sys.executable, "-m", "benchmarks.v4.baseline.tools.evaluate_baseline",
        "--config", f"benchmarks/v4/baseline/configs/{config}",
        "--phase2-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
        "--episode-manifest", manifest, "--split", "val", "--num-workers", "0", "--batch-size", "1",
        "--timestamp", stamp,
    ]
    print({"event": "starting", "config": config, "manifest": manifest, "timestamp": stamp}, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
