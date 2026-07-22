#!/usr/bin/env python3
"""Prompt-conditioned whole-slide inference with overlap-weighted stitching."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VIPS_CONCURRENCY", "1")
import pyvips
import torch

pyvips.cache_set_max(0)
pyvips.cache_set_max_mem(64 * 1024 * 1024)

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import SIZE_NAMES
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import JointPromptMaskModel, load_joint_components
from benchmarks.v4.whole_slide_inference.src.features import CellFeatureStore
from benchmarks.v4.whole_slide_inference.src.prompt_transfer import (
    EncodedRegions,
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
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/nfs-medical3/zyh/v4/whole_slide_inference"),
    )
    parser.add_argument("--timestamp")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


class SlideReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.image = pyvips.Image.new_from_file(str(path), access="random")
        if self.image.bands < 3:
            raise ValueError(f"WSI has fewer than three channels: {path}")

    def read(self, row: dict) -> np.ndarray:
        x = int(row["x_level0"]); y = int(row["y_level0"])
        width = int(row["width_level0"]); height = int(row["height_level0"])
        output_width = int(row["width_10x"]); output_height = int(row["height_10x"])
        if output_width != output_height or width != height:
            raise ValueError("J5 expects square model tiles")
        patch = self.image.crop(x, y, width, height).resize(output_width / width)
        if patch.width != output_width or patch.height != output_height:
            raise RuntimeError(
                f"libvips resize produced {(patch.width, patch.height)}, expected {(output_width, output_height)}"
            )
        patch = patch.extract_band(0, n=3)
        return np.ndarray(
            buffer=patch.write_to_memory(), dtype=np.uint8,
            shape=(output_height, output_width, 3),
        ).copy()


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[JointPromptMaskModel, dict]:
    cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    if bool(cfg["model"].get("train_parent_context", False)):
        raise ValueError("the deployed whole-slide entry point is for the formal J5 checkpoint, not Phase4/J10")
    phase2, cell, prompt, upstream = load_joint_components(
        p2cfg, args.phase2_checkpoint, args.cell_checkpoint, args.phase5_checkpoint,
        p5cfg, int(cfg["data"]["cell_feature_dim"]),
    )
    mc = cfg["model"]
    model = JointPromptMaskModel(
        phase2, cell, prompt, int(mc["region_dim"]), int(mc["graph_heads"]),
        int(mc["graph_layers"]), int(mc["graph_neighbours"]), float(mc["graph_dropout"]),
        float(mc["residual_limit"]),
        train_phase2_embedding=bool(mc.get("train_phase2_embedding", False)),
        train_phase2_assignment=bool(mc.get("train_phase2_assignment", False)),
        train_cell=bool(mc.get("train_cell", False)),
        train_prompt=bool(mc.get("train_prompt", False)),
        train_decoder=bool(mc.get("train_decoder", True)),
        train_parent_context=False,
        train_backbone_layer4=bool(mc.get("train_backbone_layer4", False)),
        num_classes=len(p2cfg["data"]["class_map"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
    )
    payload = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"J5 checkpoint mismatch missing={missing} unexpected={unexpected}")
    report = {
        "upstream": upstream,
        "joint": {"checkpoint": str(args.joint_checkpoint), "epoch": int(payload["epoch"]),
                  "missing": list(missing), "unexpected": list(unexpected)},
    }
    return model.to(device).eval(), report


def load_prompts(path: Path, width_level0: int, height_level0: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("coordinate_space") != "level0":
        raise ValueError("prompt_json.coordinate_space must be exactly 'level0'")
    size = str(value.get("prompt_size"))
    if size not in SIZE_NAMES:
        raise ValueError(f"prompt_size must be one of {SIZE_NAMES}")
    output = {"coordinate_space": "level0", "prompt_size": size}
    for sign in ("positive", "negative"):
        points = value.get(sign)
        if not isinstance(points, list) or not points:
            raise ValueError(f"prompt_json.{sign} must be a non-empty list")
        parsed = []
        for order, point in enumerate(points, start=1):
            x = float(point["x"]); y = float(point["y"])
            if not (0 <= x < width_level0 and 0 <= y < height_level0):
                raise ValueError(f"{sign} prompt {order} lies outside the WSI")
            parsed.append({"x": x, "y": y})
        output[sign] = parsed
    return output


def choose_prompt_tile(rows: list[dict], point: dict) -> dict:
    x = float(point["x"]); y = float(point["y"])
    candidates = [
        row for row in rows
        if int(row["x_level0"]) <= x < int(row["x_level0"]) + int(row["width_level0"])
        and int(row["y_level0"]) <= y < int(row["y_level0"]) + int(row["height_level0"])
    ]
    if not candidates:
        raise RuntimeError(f"prompt {(x, y)} is not covered by the tile index")
    return min(candidates, key=lambda row: (
        (x - (int(row["x_level0"]) + int(row["width_level0"]) / 2)) ** 2
        + (y - (int(row["y_level0"]) + int(row["height_level0"]) / 2)) ** 2,
        int(row["source_index"]),
    ))


def tensor_batch(
    rows: list[dict], reader: SlideReader, store: CellFeatureStore,
    max_cells: int, workers: int, device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(reader.read, rows))
    else:
        images = [reader.read(row) for row in rows]
    cell_items = []
    for row in rows:
        cells, total = store.get(int(row["source_index"]), str(row["patch_id"]))
        cell_items.append((str(row["patch_id"]), cells, total))
    packed = collate_cells(cell_items, max_cells=max_cells)
    image_tensor = torch.stack([he_to_tensor(image) for image in images]).to(device, non_blocking=True)
    packed = {key: value.to(device, non_blocking=True) for key, value in packed.items()}
    return image_tensor, packed


def subset_encoded(encoded: EncodedRegions, index: int) -> EncodedRegions:
    return EncodedRegions(*(
        value[index:index + 1]
        for value in (encoded.assignment, encoded.tokens, encoded.xy, encoded.area, encoded.active)
    ))


@torch.no_grad()
def encode_global_prompt_task(
    model: JointPromptMaskModel,
    rows: list[dict],
    prompts: dict,
    reader: SlideReader,
    store: CellFeatureStore,
    max_cells: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor] | None, list[dict], list[dict]]:
    selected = []
    for sign in ("positive", "negative"):
        for order, point in enumerate(prompts[sign], start=1):
            selected.append((sign, order, point, choose_prompt_tile(rows, point)))
    unique_rows = {int(item[3]["source_index"]): item[3] for item in selected}
    encoded_by_source = {}
    for source_index, row in unique_rows.items():
        images, cells = tensor_batch([row], reader, store, max_cells, 0, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            encoded_by_source[source_index] = encode_regions(model, images, **cells)
    records = {"positive": [], "negative": []}; keys = {"positive": set(), "negative": set()}
    tokens = {"positive": [], "negative": []}; coordinates = {"positive": [], "negative": []}
    for sign, order, point, row in selected:
        encoded = encoded_by_source[int(row["source_index"])]
        local = torch.tensor([[
            (float(point["x"]) - int(row["x_level0"])) / int(row["width_level0"]),
            (float(point["y"]) - int(row["y_level0"])) / int(row["height_level0"]),
        ]], device=device)
        slot = int(torch.cdist(local.float(), encoded.xy[0].float()).argmin())
        key = (int(row["source_index"]), slot)
        record = {
            "sign": sign, "order": order,
            "x_level0": float(point["x"]), "y_level0": float(point["y"]),
            "patch_id": str(row["patch_id"]), "source_index": int(row["source_index"]),
            "region_slot": slot,
            "region_centroid_xy_local": [float(value) for value in encoded.xy[0, slot].float().cpu()],
        }
        records[sign].append(record)
        if key not in keys[sign]:
            keys[sign].add(key)
            tokens[sign].append(encoded.tokens[0, slot])
            coordinates[sign].append(encoded.xy[0, slot])
    conflicts = keys["positive"] & keys["negative"]
    if conflicts:
        conflict_records = [
            {"source_index": source_index, "region_slot": slot}
            for source_index, slot in sorted(conflicts)
        ]
        return None, records["positive"] + records["negative"], conflict_records
    prompt_batch = {
        "positive_tokens": torch.stack(tokens["positive"])[None],
        "negative_tokens": torch.stack(tokens["negative"])[None],
        "positive_xy": torch.stack(coordinates["positive"])[None],
        "negative_xy": torch.stack(coordinates["negative"])[None],
        "positive_mask": torch.ones(1, len(tokens["positive"]), dtype=torch.bool, device=device),
        "negative_mask": torch.ones(1, len(tokens["negative"]), dtype=torch.bool, device=device),
        "prompt_size_id": torch.tensor([SIZE_NAMES.index(prompts["prompt_size"])], device=device),
    }
    with torch.autocast("cuda", dtype=amp_dtype):
        task = model.decoder.prompt_model.encode_prompt_task(prompt_batch)
    return task, records["positive"] + records["negative"], []


def write_pyramid(path: Path, array: np.ndarray, pixel_format: str) -> None:
    image = pyvips.Image.new_from_memory(
        memoryview(array), int(array.shape[1]), int(array.shape[0]), 1, pixel_format
    )
    tile_width = min(512, max(16, (int(array.shape[1]) // 16) * 16))
    tile_height = min(512, max(16, (int(array.shape[0]) // 16) * 16))
    image.tiffsave(
        str(path), tile=True, tile_width=tile_width, tile_height=tile_height, pyramid=True,
        bigtiff=True, compression="deflate",
    )


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1:
        raise ValueError("one WSI run currently requires exactly one --gpus ID")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("whole-slide J5 inference requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpus[0]); device = torch.device("cuda", args.gpus[0])
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"wsi_inference_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = pd.read_parquet(args.tile_index).sort_values("source_index").to_dict("records")
    width, height, downsample = validate_tile_rows(rows)
    wsi_path = Path(rows[0]["wsi_path"])
    reader = SlideReader(wsi_path)
    source_width_10x = reader.image.width // downsample
    source_height_10x = reader.image.height // downsample
    if (width, height) != (source_width_10x, source_height_10x):
        raise RuntimeError(
            "tile index is not a whole-slide canvas: "
            f"index={(width, height)} source={(source_width_10x, source_height_10x)}"
        )
    prompts = load_prompts(args.prompt_json, reader.image.width, reader.image.height)
    inputs = [
        args.config, args.phase2_config, args.phase5_config, args.phase2_checkpoint,
        args.cell_checkpoint, args.phase5_checkpoint, args.joint_checkpoint,
        args.tile_index, args.cell_feature_manifest, args.prompt_json,
    ]
    run_manifest = {
        "timestamp": stamp, "status": "started", "split": str(rows[0]["split"]),
        "test_used": str(rows[0]["split"]) == "test",
        "wsi_id": str(rows[0]["wsi_id"]), "wsi_path": str(wsi_path),
        "tile_count": len(rows), "output_shape_10x": [height, width],
        "level0_downsample": downsample, "gpus": args.gpus,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "seed": args.seed, "threshold": args.threshold, "amp_dtype": args.amp_dtype,
        "inputs": {str(path): sha256(path) for path in inputs},
    }
    save_json(output / "run_manifest.json", run_manifest)
    store = CellFeatureStore(args.cell_feature_manifest, len(rows))
    model, load_report = load_model(args, device)
    max_cells = int(load_config(args.config)["data"]["max_cells_per_patch"])
    prompt_task, prompt_records, conflicts = encode_global_prompt_task(
        model, rows, prompts, reader, store, max_cells, device, amp_dtype,
    )
    if conflicts:
        metadata = {
            **run_manifest, "status": "abstained_prompt_conflict", "mask_returned": False,
            "prompt_records": prompt_records, "conflicts": conflicts, "load_report": load_report,
        }
        save_json(output / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2), flush=True)
        return
    accumulator = OverlapAccumulator(output, height, width, args.threshold)
    tile_height = int(rows[0]["height_10x"]); tile_width = int(rows[0]["width_10x"])
    window = blend_window(tile_height, tile_width)
    completed_path = output / "completed.jsonl"; failures_path = output / "failures.jsonl"
    with completed_path.open("a", encoding="utf-8") as completed, failures_path.open("a", encoding="utf-8") as failures:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            try:
                images, cells = tensor_batch(
                    batch_rows, reader, store, max_cells, args.num_workers, device,
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                    encoded = encode_regions(model, images, **cells)
                    result = decode_regions_with_task(model, encoded, prompt_task)
                probabilities = result["pixel_probability"].float().cpu().numpy()
                for row, probability in zip(batch_rows, probabilities, strict=True):
                    accumulator.add(probability, int(row["x_10x"]), int(row["y_10x"]), window)
                    completed.write(json.dumps({
                        "source_index": int(row["source_index"]), "patch_id": str(row["patch_id"]),
                        "probability_min": float(probability.min()),
                        "probability_max": float(probability.max()),
                    }) + "\n")
                completed.flush()
            except Exception as exc:
                failures.write(json.dumps({"start": start, "error": repr(exc)}) + "\n")
                failures.flush()
                raise
            print({"event": "tiles_complete", "done": start + len(batch_rows), "total": len(rows)}, flush=True)
    probability, mask = accumulator.finalize()
    probability_tiff = output / "probability_10x_pyramid.tif"
    mask_tiff = output / "mask_10x_pyramid.tif"
    coverage_tiff = output / "blend_weight_10x_pyramid.tif"
    write_pyramid(probability_tiff, probability, "float")
    write_pyramid(mask_tiff, mask, "uchar")
    write_pyramid(coverage_tiff, accumulator.weight_sum, "float")
    metadata = {
        **run_manifest, "status": "complete", "mask_returned": True,
        "prompt_records": prompt_records, "conflicts": [], "load_report": load_report,
        "probability_tiff": str(probability_tiff), "mask_tiff": str(mask_tiff),
        "coverage_tiff": str(coverage_tiff),
        "probability_range": [float(probability.min()), float(probability.max())],
        "positive_fraction_at_threshold": float((mask > 0).mean()),
        "coordinate_contract": {
            "prompt_input": "level0 pixels", "output": "10x pixels",
            "level0_to_output": f"floor(level0 / {downsample})",
        },
        "scope": "one complete WSI; formal J5 without Phase4 parent context",
    }
    save_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
