#!/usr/bin/env python3
"""Run one promptable baseline patchwise and stitch exact whole-slide predictions."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import yaml

from benchmarks.v4.baseline.adapters import EpisodeRequest, build_adapter
from benchmarks.v4.baseline.integrations.sam_external import read_he_patch_white_padded
from benchmarks.v4.whole_slide_inference.infer_wsi import write_pyramid
from benchmarks.v4.whole_slide_inference.src.tiling import (
    OverlapAccumulator,
    blend_window,
    validate_tile_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument(
        "--class-config",
        type=Path,
        default=Path("benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml"),
    )
    parser.add_argument("--wsi-id", action="append")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-calls", type=int)
    parser.add_argument(
        "--context-scale",
        type=int,
        default=1,
        help=(
            "centered inference field relative to the stitched tile; context-scale=2 "
            "runs on a 2x field and stitches only the central tile"
        ),
    )
    parser.add_argument(
        "--context-prompt-mode",
        choices=("central_tile", "all_in_context"),
        default="central_tile",
        help=(
            "central_tile preserves the historical independent tile prompt; "
            "all_in_context projects the frozen global click set into every context"
        ),
    )
    return parser.parse_args()


def exact_metrics(gt_path: Path, mask: np.ndarray, target_rgb: tuple[int, int, int]) -> dict:
    gt = pyvips.Image.new_from_file(str(gt_path), access="sequential").extract_band(0, n=3)
    if gt.width != mask.shape[1] or gt.height != mask.shape[0]:
        if mask.shape[1] - gt.width not in (0, 1) or mask.shape[0] - gt.height not in (0, 1):
            raise ValueError(f"GT/mask extent mismatch: GT={(gt.width, gt.height)} mask={mask.shape[::-1]}")
    pred_count = intersection = gt_count = 0
    for y in range(0, gt.height, 1024):
        height = min(1024, gt.height - y)
        crop = gt.crop(0, y, gt.width, height)
        target = (
            (crop[0] == target_rgb[0])
            & (crop[1] == target_rgb[1])
            & (crop[2] == target_rgb[2])
        )
        truth = np.frombuffer(target.cast("uchar").write_to_memory(), np.uint8).reshape(height, gt.width) > 0
        pred = np.asarray(mask[y:y + height, :gt.width]) > 0
        gt_count += int(truth.sum())
        pred_count += int(pred.sum())
        intersection += int((truth & pred).sum())
    union = gt_count + pred_count - intersection
    return {
        "gt_positive_pixels": gt_count,
        "prediction_positive_pixels": pred_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "dice": 2 * intersection / (gt_count + pred_count) if gt_count + pred_count else 1.0,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_count if pred_count else float(gt_count == 0),
        "recall": intersection / gt_count if gt_count else float(pred_count == 0),
    }


def parse_point(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([json.loads(str(value))], dtype=np.float32)


def collect_global_points(frame: pd.DataFrame, allowed: set[int], column: str) -> np.ndarray:
    points = []
    for row in frame.loc[frame["source_index"].astype(int).isin(allowed)].to_dict("records"):
        local = parse_point(row[column])
        if len(local):
            points.append(local[0] + np.asarray([row["x_10x"], row["y_10x"]], dtype=np.float32))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def points_in_context(
    points_global: np.ndarray,
    context_x_10x: int,
    context_y_10x: int,
    context_size: int,
) -> np.ndarray:
    if not len(points_global):
        return np.empty((0, 2), dtype=np.float32)
    origin = np.asarray([context_x_10x, context_y_10x], dtype=np.float32)
    local = points_global - origin
    keep = (
        (local[:, 0] >= 0) & (local[:, 0] < context_size)
        & (local[:, 1] >= 0) & (local[:, 1] < context_size)
    )
    return local[keep]


def main() -> None:
    args = parse_args()
    if args.context_scale < 1:
        raise ValueError("context-scale must be a positive integer")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    class_config = yaml.safe_load(args.class_config.read_text(encoding="utf-8"))
    class_by_id = {
        int(item["id"]): (str(item["name"]), tuple(int(value) for value in item["rgb"]))
        for item in class_config["data"]["class_map"]
    }
    config["device"] = f"cuda:{args.gpu}"
    tasks = pd.read_parquet(args.task_manifest)
    if args.wsi_id:
        tasks = tasks.loc[tasks["wsi_id"].isin(args.wsi_id)]
    if tasks.empty:
        raise ValueError("no requested WSI tasks")
    root = args.output_root / f"{config['name']}_gt_guided_wsi_{args.timestamp}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    adapter = build_adapter(config)
    summaries = []
    for task in tasks.to_dict("records"):
        wsi_id = str(task["wsi_id"])
        target_class = int(task["target_class"])
        if target_class not in class_by_id:
            raise ValueError(f"target class {target_class} is absent from {args.class_config}")
        target_name, target_rgb = class_by_id[target_class]
        if str(task["target_class_name"]) != target_name:
            raise ValueError(
                f"target class contract mismatch: task={task['target_class_name']} config={target_name}"
            )
        frame = pd.read_parquet(task["patch_prompt_manifest"]).sort_values("source_index").reset_index(drop=True)
        width, height, _ = validate_tile_rows(frame.to_dict("records"))
        called = frame.loc[frame["has_target"]]
        if (called["target_pixels"] <= 0).any() or not called["positive_audit"].all():
            raise RuntimeError(f"invalid positive prompt contract for {wsi_id}")
        if not called["negative_audit"].all():
            raise RuntimeError(f"invalid negative prompt contract for {wsi_id}")
        if args.limit_calls is not None:
            allowed = set(called.iloc[:args.limit_calls]["source_index"].astype(int))
        else:
            allowed = set(called["source_index"].astype(int))
        global_positive = collect_global_points(frame, allowed, "positive_point_10x")
        global_negative = collect_global_points(frame, allowed, "negative_point_10x")
        out = root / wsi_id
        out.mkdir()
        accumulator = OverlapAccumulator(out, height, width, args.threshold)
        tile_height = int(frame.iloc[0]["height_10x"])
        tile_width = int(frame.iloc[0]["width_10x"])
        if tile_height != tile_width:
            raise ValueError(f"context protocol requires square tiles, got {(tile_width, tile_height)}")
        context_size = tile_width * args.context_scale
        context_offset = (context_size - tile_width) // 2
        if 2 * context_offset + tile_width != context_size:
            raise ValueError(
                f"context-scale={args.context_scale} does not center an integer crop for tile size {tile_width}"
            )
        downsample = int(frame.iloc[0]["level0_downsample"])
        window = blend_window(tile_height, tile_width)
        zero = np.zeros(window.shape, dtype=np.float32)
        completed_path = out / "completed.jsonl"
        started = time.perf_counter()
        model_calls = positive_prompt_instances = negative_prompt_instances = 0
        with completed_path.open("a", encoding="utf-8") as completed:
            for row in frame.to_dict("records"):
                source_index = int(row["source_index"])
                probability = zero
                status = "skipped_no_target"
                latency = 0.0
                if bool(row["has_target"]) and source_index in allowed:
                    context_x_level0 = int(row["x_level0"]) - context_offset * downsample
                    context_y_level0 = int(row["y_level0"]) - context_offset * downsample
                    context_width_level0 = context_size * downsample
                    image = read_he_patch_white_padded(
                        Path(row["wsi_path"]), context_x_level0, context_y_level0,
                        context_width_level0, context_width_level0, context_size,
                    )
                    if args.context_prompt_mode == "central_tile":
                        positive = parse_point(row["positive_point_10x"]) + context_offset
                        negative = parse_point(row["negative_point_10x"]) + context_offset
                    else:
                        context_x_10x = int(row["x_10x"]) - context_offset
                        context_y_10x = int(row["y_10x"]) - context_offset
                        positive = points_in_context(
                            global_positive, context_x_10x, context_y_10x, context_size,
                        )
                        negative = points_in_context(
                            global_negative, context_x_10x, context_y_10x, context_size,
                        )
                        if not len(positive):
                            raise RuntimeError(
                                f"{wsi_id} {row['patch_id']} has no positive prompt in its context"
                            )
                    request = EpisodeRequest(
                        occurrence_id=str(row["patch_id"]),
                        image=np.asarray(image, dtype=np.uint8),
                        positive_points=positive,
                        negative_points=negative,
                        prompt_type="point",
                        multiscale_inputs={
                            "fine": np.asarray(image, dtype=np.uint8),
                            "wsi_path": str(row["wsi_path"]),
                            "center_level0": [
                                int(row["x_level0"]) + int(row["width_level0"]) / 2,
                                int(row["y_level0"]) + int(row["height_level0"]) / 2,
                            ],
                            "fine_box_level0": [
                                context_x_level0, context_y_level0,
                                context_width_level0, context_width_level0,
                            ],
                        } if bool(config.get("requires_multiscale_inputs", False)) else None,
                    )
                    prediction = adapter.timed_predict(request)
                    if prediction.status != "completed":
                        raise RuntimeError(f"{wsi_id} {row['patch_id']} returned {prediction.status}")
                    context_probability = np.asarray(prediction.probability, dtype=np.float32)
                    if context_probability.shape != (context_size, context_size):
                        raise RuntimeError(
                            f"{wsi_id} {row['patch_id']} returned context shape "
                            f"{context_probability.shape}, expected {(context_size, context_size)}"
                        )
                    probability = context_probability[
                        context_offset:context_offset + tile_height,
                        context_offset:context_offset + tile_width,
                    ]
                    status = "completed"
                    latency = float(prediction.latency_ms)
                    model_calls += 1
                    positive_prompt_instances += len(positive)
                    negative_prompt_instances += len(negative)
                accumulator.add(probability, int(row["x_10x"]), int(row["y_10x"]), window)
                completed.write(json.dumps({
                    "source_index": source_index, "patch_id": str(row["patch_id"]),
                    "status": status, "latency_ms": latency,
                }) + "\n")
                completed.flush()
                if source_index % 25 == 0:
                    print(json.dumps({
                        "event": "progress", "method": config["name"], "wsi_id": wsi_id,
                        "tiles_done": source_index + 1, "tiles_total": len(frame),
                        "model_calls": model_calls,
                    }), flush=True)
        probability, mask = accumulator.finalize()
        probability_tiff = out / "probability_10x_pyramid.tif"
        mask_tiff = out / "mask_10x_pyramid.tif"
        write_pyramid(probability_tiff, probability, "float")
        write_pyramid(mask_tiff, mask, "uchar")
        metrics = exact_metrics(Path(task["gt_path"]), mask, target_rgb)
        summary = {
            "method": str(config["name"]), "wsi_id": wsi_id,
            "target_class": target_class, "target_class_name": target_name,
            "target_rgb": list(target_rgb), **metrics,
            "patches": len(frame), "target_patches": int(frame["has_target"].sum()),
            "model_calls": model_calls,
            "positive_clicks": int(len(global_positive)),
            "negative_clicks": int(len(global_negative)),
            "positive_prompt_instances": positive_prompt_instances,
            "negative_prompt_instances": negative_prompt_instances,
            "context_scale": args.context_scale,
            "context_prompt_mode": args.context_prompt_mode,
            "context_size_10x": context_size,
            "stitched_tile_size_10x": tile_width,
            "context_crop_offset_10x": context_offset,
            "elapsed_seconds": time.perf_counter() - started,
            "mask_tiff": str(mask_tiff), "probability_tiff": str(probability_tiff),
            "complete_protocol": args.limit_calls is None,
        }
        (out / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps({"event": "wsi_complete", **summary}), flush=True)
    pd.DataFrame(summaries).to_parquet(root / f"wsi_metrics_{args.timestamp}.parquet", index=False)
    (root / "metadata.json").write_text(json.dumps({
        "status": "complete", "method": config["name"], "timestamp": args.timestamp,
        "context_scale": args.context_scale,
        "context_prompt_mode": args.context_prompt_mode,
        "wsi_count": len(summaries), "summaries": summaries,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(root), "wsi_count": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
