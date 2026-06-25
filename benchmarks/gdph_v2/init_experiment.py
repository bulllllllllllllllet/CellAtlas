from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT, initialize_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the GDPH v2 benchmark.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--mapping_csv", default="benchmarks/tissue_seg_runs/dataset_mapping.csv"
    )
    parser.add_argument(
        "--palette_json", default="benchmarks/tissue_seg_runs/gdph_tissue_palette.json"
    )
    parser.add_argument(
        "--xcell_checkpoint",
        default="experiments/crc012_holdout3/20260515_160711/he_model_best.pth",
    )
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--cellpose_checkpoint", default="/home/zyh/.cellpose/models/cpsam")
    args = parser.parse_args()

    config = initialize_experiment(
        output_root=args.output_root,
        mapping_csv=args.mapping_csv,
        palette_json=args.palette_json,
        xcell_checkpoint=args.xcell_checkpoint,
        ctranspath_checkpoint=args.ctranspath_checkpoint,
        cellpose_checkpoint=args.cellpose_checkpoint,
        repo_root=Path.cwd(),
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

