#!/usr/bin/env python3
"""Smoke-test 3+1 and 5+1 frozen prompts for each external baseline."""
from __future__ import annotations

import subprocess
import sys


ROOT = "/home/zhaoyh/CellAtlas"
MANIFESTS = {
    "3p1n": "/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet",
    "5p1n": "/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet",
}
METHODS = ("sam_zero_shot.yaml", "sam_med2d_zero_shot.yaml", "wsi_sam_zero_shot.yaml")
for method_index, config in enumerate(METHODS):
    for budget_index, (budget, manifest) in enumerate(MANIFESTS.items()):
        stamp = f"20260724_1954{method_index}{budget_index}"
        command = [
            sys.executable, "-m", "benchmarks.v4.baseline.tools.evaluate_baseline",
            "--config", f"benchmarks/v4/baseline/configs/{config}",
            "--phase2-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
            "--episode-manifest", manifest, "--split", "val", "--limit", "1",
            "--num-workers", "0", "--batch-size", "1", "--timestamp", stamp,
        ]
        print({"event": "smoke_start", "config": config, "budget": budget, "timestamp": stamp}, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
