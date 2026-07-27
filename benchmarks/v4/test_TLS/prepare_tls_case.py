#!/usr/bin/env python3
"""Parse KFB annotations, build a 10x tile index, prompts, and TLS ground truth."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import pyvips
from scipy.ndimage import distance_transform_edt

from benchmarks.v4.whole_slide_inference.src.tiling import build_tile_rows, validate_tile_rows
from module.KFBreader.kfbreader import KFBSlide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--wsi-id", default="HZ_TLS_202002827_01304707")
    parser.add_argument("--positive-polygons", type=int, nargs="+", default=[0, 4, 5])
    parser.add_argument("--negative-count", type=int, default=3)
    parser.add_argument("--level0-downsample", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/test_TLS"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def load_polygons(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("annotation JSON must be a non-empty list")
    polygons = []
    for index, item in enumerate(payload):
        if item.get("type") != "Curve" or not isinstance(item.get("points"), list):
            raise ValueError(f"annotation {index} is not a Curve with points")
        polygon = np.asarray([[float(point["x"]), float(point["y"])] for point in item["points"]], np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise ValueError(f"invalid polygon {index}: {polygon.shape}")
        polygons.append(polygon)
    return polygons


def interior_point(polygon: np.ndarray) -> tuple[float, float]:
    lower = np.floor(polygon.min(0)).astype(int)
    upper = np.ceil(polygon.max(0)).astype(int)
    shape = (int(upper[1] - lower[1] + 3), int(upper[0] - lower[0] + 3))
    mask = np.zeros(shape, np.uint8)
    local = np.rint(polygon - lower + 1).astype(np.int32)
    cv2.fillPoly(mask, [local], 1)
    distance = distance_transform_edt(mask)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    if distance[y, x] <= 0:
        raise RuntimeError("polygon has no rasterized interior")
    return float(x + lower[0] - 1), float(y + lower[1] - 1)


def choose_negative_points(
    slide: KFBSlide, polygons: list[np.ndarray], count: int
) -> tuple[list[dict[str, float]], np.ndarray]:
    level = slide.level_count - 1
    downsample = float(slide.level_downsamples[level])
    width, height = slide.level_dimensions[level]
    thumbnail = np.asarray(slide.read_region((0, 0), level, (width, height)), dtype=np.uint8)[..., :3]
    maximum = thumbnail.max(2).astype(np.int16)
    minimum = thumbnail.min(2).astype(np.int16)
    tissue = (thumbnail.mean(2) < 235) & ((maximum - minimum) > 8)
    annotated = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(annotated, [np.rint(polygon / downsample).astype(np.int32)], 1)
    exclusion_radius = max(8, int(round(2048 / downsample)))
    excluded = cv2.dilate(annotated, np.ones((exclusion_radius, exclusion_radius), np.uint8))
    candidates = tissue & (excluded == 0)
    if int(candidates.sum()) < count:
        raise RuntimeError(f"only {int(candidates.sum())} negative tissue candidates")
    selected = []
    available = candidates.astype(np.uint8)
    distance_from_tls = distance_transform_edt(annotated == 0)
    for _ in range(count):
        score = np.where(available > 0, distance_from_tls, -1)
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        selected.append({"x": float((x + 0.5) * downsample), "y": float((y + 0.5) * downsample)})
        cv2.circle(available, (int(x), int(y)), max(16, int(round(4096 / downsample))), 0, -1)
    return selected, thumbnail


def write_mask_tiff(path: Path, mask: np.ndarray) -> None:
    image = pyvips.Image.new_from_memory(memoryview(mask), mask.shape[1], mask.shape[0], 1, "uchar")
    image.tiffsave(
        str(path), tile=True, tile_width=512, tile_height=512, pyramid=True,
        bigtiff=True, compression="deflate",
    )


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"tls_case_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    if not args.wsi_path.is_file() or not args.annotation_json.is_file():
        raise FileNotFoundError("WSI or annotation JSON does not exist")
    output.mkdir(parents=True)
    polygons = load_polygons(args.annotation_json)
    if len(set(args.positive_polygons)) != len(args.positive_polygons):
        raise ValueError("positive polygon indices must be unique")
    if any(index < 0 or index >= len(polygons) for index in args.positive_polygons):
        raise IndexError("positive polygon index outside annotation list")

    slide = KFBSlide(str(args.wsi_path))
    width_level0, height_level0 = slide.dimensions
    if any(np.any(poly[:, 0] < 0) or np.any(poly[:, 0] >= width_level0)
           or np.any(poly[:, 1] < 0) or np.any(poly[:, 1] >= height_level0) for poly in polygons):
        raise ValueError("annotation polygon lies outside WSI level-0 bounds")
    negatives, thumbnail = choose_negative_points(slide, polygons, args.negative_count)
    rows = build_tile_rows(
        args.wsi_id, str(args.wsi_path), width_level0, height_level0,
        args.level0_downsample, args.tile_size, args.stride, "val",
    )
    width_10x, height_10x, downsample = validate_tile_rows(rows)
    tile_index = output / "wsi_tile_index.parquet"
    pd.DataFrame(rows).to_parquet(tile_index, index=False)

    positives = [
        {"polygon_index": int(index), **dict(zip(("x", "y"), interior_point(polygons[index])))}
        for index in args.positive_polygons
    ]
    prompts = {
        "coordinate_space": "level0", "prompt_size": "large",
        "positive": [{"x": item["x"], "y": item["y"]} for item in positives],
        "negative": negatives,
    }
    prompt_path = output / "prompts.json"
    prompt_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    gt = np.zeros((height_10x, width_10x), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(gt, [np.rint(polygon / downsample).astype(np.int32)], 255)
    gt_path = output / "tls_gt_10x_pyramid.tif"
    write_mask_tiff(gt_path, gt)

    preview = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    preview_downsample = float(slide.level_downsamples[slide.level_count - 1])
    for index, polygon in enumerate(polygons):
        color = (0, 255, 0) if index in args.positive_polygons else (0, 255, 255)
        cv2.polylines(preview, [np.rint(polygon / preview_downsample).astype(np.int32)], True, color, 3)
    for point in positives:
        cv2.drawMarker(preview, (round(point["x"] / preview_downsample), round(point["y"] / preview_downsample)), (0, 255, 0), cv2.MARKER_CROSS, 18, 3)
    for point in negatives:
        cv2.drawMarker(preview, (round(point["x"] / preview_downsample), round(point["y"] / preview_downsample)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    preview_path = output / "annotation_prompt_preview.png"
    cv2.imwrite(str(preview_path), preview)

    summary = {
        "timestamp": stamp, "wsi_path": str(args.wsi_path), "annotation_json": str(args.annotation_json),
        "wsi_dimensions_level0": [width_level0, height_level0],
        "level_dimensions": [list(value) for value in slide.level_dimensions],
        "level_downsamples": list(slide.level_downsamples), "level0_downsample": downsample,
        "output_shape_10x": [height_10x, width_10x], "tile_count": len(rows),
        "annotation_count": len(polygons), "positive_polygon_indices": args.positive_polygons,
        "held_out_polygon_indices": sorted(set(range(len(polygons))) - set(args.positive_polygons)),
        "positive_prompts": positives, "negative_prompts": negatives,
        "tile_index": str(tile_index), "prompt_json": str(prompt_path), "gt_mask": str(gt_path),
        "preview": str(preview_path), "visual_review": "pending_user_review",
    }
    (output / "case_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
