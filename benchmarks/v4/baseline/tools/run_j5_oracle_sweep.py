#!/usr/bin/env python3
"""Run the remaining fixed five-WSI J5 best-of-K oracle candidates sequentially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


CELL_FEATURES = {
    "1033746-12-HE-DX1": "/nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113000/feature_index.parquet",
    "1028417-R1-HE-DX1": "/nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113001/feature_index.parquet",
    "1321593-10-HE-DX1": "/nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_113002/feature_index.parquet",
    "1416664-10-HE-DX1": "/nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_114917/feature_index.parquet",
    "1504774-12-HE-DX1": "/nfs-medical3/zyh/v4/whole_slide_inference/phase3_features/xcell_features_val_20260726_114918/feature_index.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--skip", action="append", default=[], help="wsi_id:candidate_index")
    parser.add_argument("--include-zero", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root / f"j5_wsi_oracle_sweep_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    candidates = pd.read_parquet(args.candidate_manifest).sort_values(
        ["wsi_id", "candidate_index"]
    )
    skip = set(args.skip)
    if not args.include_zero:
        candidates = candidates.loc[candidates["candidate_index"] > 0]
    candidates = candidates.loc[
        ~candidates.apply(lambda row: f"{row.wsi_id}:{int(row.candidate_index)}" in skip, axis=1)
    ].reset_index(drop=True)
    completed = output / "completed.jsonl"
    base_time = datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    for run_index, row in enumerate(candidates.itertuples(index=False), start=1):
        stamp = (base_time + timedelta(seconds=run_index)).strftime("%Y%m%d_%H%M%S")
        inference_dir = Path("/nfs-medical3/zyh/v4/whole_slide_inference") / f"wsi_inference_{stamp}"
        infer = [
            sys.executable, "-m", "benchmarks.v4.whole_slide_inference.infer_wsi",
            "--config", "benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml",
            "--phase2-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
            "--phase5-config", "benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml",
            "--phase2-checkpoint", "/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth",
            "--cell-checkpoint", "/nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth",
            "--phase5-checkpoint", "/nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth",
            "--joint-checkpoint", "/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth",
            "--tile-index", str(row.tile_index),
            "--cell-feature-manifest", CELL_FEATURES[str(row.wsi_id)],
            "--prompt-json", str(row.prompt_json),
            "--gpus", str(args.gpu), "--batch-size", "2", "--num-workers", "4",
            "--output-root", "/nfs-medical3/zyh/v4/whole_slide_inference",
            "--timestamp", stamp,
        ]
        print(json.dumps({"event": "candidate_start", "wsi_id": row.wsi_id,
                          "candidate_index": int(row.candidate_index), "timestamp": stamp}), flush=True)
        subprocess.run(infer, check=True)
        visual_stamp = (base_time + timedelta(minutes=10, seconds=run_index)).strftime(
            "%Y%m%d_%H%M%S"
        )
        evaluate = [
            sys.executable, "-m", "benchmarks.v4.whole_slide_inference.visualize_wsi_gt_prediction",
            "--inference-dir", str(inference_dir), "--gt-path", str(row.gt_path),
            "--he-path", str(row.wsi_path),
            "--class-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
            "--target-class", str(getattr(row, "target_class_name", "muscle")),
            "--timestamp", visual_stamp, "--preview-width", "1600",
        ]
        subprocess.run(evaluate, check=True)
        report_path = inference_dir / f"wsi_gt_vs_prediction_{visual_stamp}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        record = {
            "wsi_id": str(row.wsi_id), "candidate_index": int(row.candidate_index),
            "timestamp": stamp, "dice": float(report["metrics"]["dice"]),
            "iou": float(report["metrics"]["iou"]), "inference_dir": str(inference_dir),
            "report": str(report_path),
        }
        with completed.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"event": "candidate_complete", **record}), flush=True)
    (output / "metadata.json").write_text(json.dumps({
        "status": "complete", "timestamp": args.timestamp, "new_runs": len(candidates),
        "candidate_manifest": str(args.candidate_manifest), "completed": str(completed),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
