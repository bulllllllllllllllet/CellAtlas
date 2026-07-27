#!/usr/bin/env python3
"""Encode one WSI tile batch once and decode multiple J5 prompt tasks."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.whole_slide_inference.infer_wsi import (
    SlideReader,
    encode_global_prompt_task,
    load_model,
    load_prompts,
    save_json,
    tensor_batch,
    write_pyramid,
)
from benchmarks.v4.whole_slide_inference.src.features import CellFeatureStore
from benchmarks.v4.whole_slide_inference.src.prompt_transfer import (
    decode_regions_with_task,
    encode_regions,
)
from benchmarks.v4.whole_slide_inference.src.tiling import (
    OverlapAccumulator,
    blend_window,
    validate_tile_rows,
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
    parser.add_argument("--tile-index", type=Path, required=True)
    parser.add_argument("--cell-feature-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--wsi-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("multi-prompt J5 inference requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    output = args.output_root / f"j5_multi_prompt_wsi_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    candidate_frame = pd.read_parquet(args.candidate_manifest)
    candidate_frame = candidate_frame.loc[candidate_frame["wsi_id"].eq(args.wsi_id)].sort_values(
        "candidate_index"
    )
    if candidate_frame.empty or candidate_frame["candidate_index"].duplicated().any():
        raise ValueError("candidate manifest must contain unique candidates for requested WSI")

    rows = pd.read_parquet(args.tile_index).sort_values("source_index").to_dict("records")
    width, height, downsample = validate_tile_rows(rows)
    if str(rows[0]["wsi_id"]) != args.wsi_id:
        raise ValueError("tile-index WSI differs from requested WSI")
    reader = SlideReader(Path(rows[0]["wsi_path"]))
    store = CellFeatureStore(args.cell_feature_manifest, len(rows))
    model, load_report = load_model(args, device)
    max_cells = int(load_config(args.config)["data"]["max_cells_per_patch"])

    tasks = []
    for candidate in candidate_frame.itertuples(index=False):
        prompts = load_prompts(Path(candidate.prompt_json), reader.width, reader.height)
        task, records, conflicts = encode_global_prompt_task(
            model, rows, prompts, reader, store, max_cells, device, amp_dtype
        )
        if conflicts:
            candidate_dir = output / f"candidate_{int(candidate.candidate_index):02d}"
            candidate_dir.mkdir()
            save_json(candidate_dir / "metadata.json", {
                "status": "abstained_prompt_conflict",
                "mode": "batched_multi_prompt",
                "wsi_id": args.wsi_id,
                "candidate_index": int(candidate.candidate_index),
                "prompt_json": str(candidate.prompt_json),
                "prompt_records": records,
                "conflicts": conflicts,
                "mask_returned": False,
            })
            print(json.dumps({
                "event": "candidate_abstained_prompt_conflict",
                "candidate_index": int(candidate.candidate_index),
                "conflicts": conflicts,
            }), flush=True)
            continue
        candidate_dir = output / f"candidate_{int(candidate.candidate_index):02d}"
        candidate_dir.mkdir()
        tasks.append({
            "candidate_index": int(candidate.candidate_index),
            "prompt_json": str(candidate.prompt_json),
            "task": task,
            "prompt_records": records,
            "output": candidate_dir,
            "accumulator": OverlapAccumulator(candidate_dir, height, width, args.threshold),
        })

    window = blend_window(int(rows[0]["height_10x"]), int(rows[0]["width_10x"]))
    completed = output / "completed.jsonl"
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        images, cells = tensor_batch(
            batch_rows, reader, store, max_cells, args.num_workers, device
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            encoded = encode_regions(model, images, **cells)
            for candidate in tasks:
                result = decode_regions_with_task(model, encoded, candidate["task"])
                probabilities = result["pixel_probability"].float().cpu().numpy()
                for row, probability in zip(batch_rows, probabilities, strict=True):
                    candidate["accumulator"].add(
                        probability, int(row["x_10x"]), int(row["y_10x"]), window
                    )
        with completed.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "start": start, "stop": start + len(batch_rows), "total": len(rows)
            }) + "\n")
        print(json.dumps({
            "event": "tiles_complete", "done": start + len(batch_rows), "total": len(rows),
            "candidate_count": len(tasks),
        }), flush=True)

    summaries = []
    for candidate in tasks:
        probability, mask = candidate["accumulator"].finalize()
        probability_tiff = candidate["output"] / "probability_10x_pyramid.tif"
        mask_tiff = candidate["output"] / "mask_10x_pyramid.tif"
        write_pyramid(probability_tiff, probability, "float")
        write_pyramid(mask_tiff, mask, "uchar")
        metadata = {
            "status": "complete", "mode": "batched_multi_prompt",
            "wsi_id": args.wsi_id, "candidate_index": candidate["candidate_index"],
            "prompt_json": candidate["prompt_json"],
            "prompt_records": candidate["prompt_records"],
            "output_shape_10x": [height, width], "level0_downsample": downsample,
            "threshold": args.threshold, "mask_tiff": str(mask_tiff),
            "probability_tiff": str(probability_tiff), "load_report": load_report,
            "shared_tile_encoding": True, "candidate_count": len(tasks),
        }
        save_json(candidate["output"] / "metadata.json", metadata)
        summaries.append(metadata)
    save_json(output / "metadata.json", {
        "status": "complete", "mode": "batched_multi_prompt", "wsi_id": args.wsi_id,
        "requested_candidate_count": len(candidate_frame),
        "decoded_candidate_count": len(tasks), "tile_count": len(rows), "candidates": summaries,
    })
    print(json.dumps({"output": str(output), "candidates": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
