from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import PRET_DIR, read_csv, safe_l2_normalize, weighted_concat, write_json_atomic


OUTPUT_VARIANT = "image_cell_reg_texture_cellstats"


def _segment_moments(values: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    area = np.bincount(segment_ids, minlength=num_segments).astype(np.float32)
    sums = np.bincount(segment_ids, weights=values, minlength=num_segments)
    sq_sums = np.bincount(segment_ids, weights=values * values, minlength=num_segments)
    mean = sums / np.maximum(area, 1.0)
    var = np.maximum(sq_sums / np.maximum(area, 1.0) - mean * mean, 0.0)
    return np.stack([mean, np.sqrt(var)], axis=1).astype(np.float32)


def _segment_percentiles(values: np.ndarray, segment_ids: np.ndarray, num_segments: int, percentiles: tuple[int, ...]) -> np.ndarray:
    output = np.zeros((num_segments, len(percentiles)), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(segment_ids, kind="stable")
    sorted_segments = segment_ids[order]
    sorted_values = values[order]
    starts = np.searchsorted(sorted_segments, np.arange(num_segments), side="left")
    ends = np.searchsorted(sorted_segments, np.arange(num_segments), side="right")
    for segment_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end > start:
            output[segment_id] = np.percentile(sorted_values[start:end], percentiles)
    return output


def _resize_for_texture(rgb: np.ndarray, segments: np.ndarray, max_dimension: int) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = segments.shape
    if max_dimension <= 0 or max(height, width) <= max_dimension:
        return rgb, segments, 1.0
    scale = max_dimension / max(height, width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    small_rgb = np.asarray(Image.fromarray(rgb).resize(new_size, Image.BILINEAR), dtype=np.uint8)
    small_segments = np.asarray(Image.fromarray(segments.astype(np.int32), mode="I").resize(new_size, Image.NEAREST), dtype=np.int32)
    return small_rgb, small_segments, scale


def _texture_block(rgb: np.ndarray, segments: np.ndarray, num_segments: int, max_dimension: int) -> tuple[np.ndarray, float]:
    from skimage.color import rgb2gray, rgb2hed
    from skimage.feature import local_binary_pattern
    from skimage.filters import sobel

    rgb, segments, scale = _resize_for_texture(rgb, segments, max_dimension)
    valid = segments >= 0
    segment_ids = segments[valid].reshape(-1).astype(np.int64)
    rgb_float = rgb.astype(np.float32) / 255.0
    gray = rgb2gray(rgb_float).astype(np.float32)
    hed = rgb2hed(rgb_float).astype(np.float32)
    edge = sobel(gray).astype(np.float32)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform").astype(np.float32)

    parts = []
    for channel in range(3):
        parts.append(_segment_moments(hed[:, :, channel][valid], segment_ids, num_segments))
        parts.append(_segment_percentiles(hed[:, :, channel][valid], segment_ids, num_segments, (10, 50, 90)))
    for image in (gray, edge, lbp):
        parts.append(_segment_moments(image[valid], segment_ids, num_segments))
        parts.append(_segment_percentiles(image[valid], segment_ids, num_segments, (10, 50, 90)))
    return np.concatenate(parts, axis=1).astype(np.float32), scale


def _cellstats_block(records: list[dict[str, str]]) -> np.ndarray:
    count = np.asarray([float(row.get("cell_count", 0.0)) for row in records], dtype=np.float32)
    density = np.asarray([float(row.get("cell_density", 0.0)) for row in records], dtype=np.float32)
    area = np.asarray([float(row.get("area_10x_pixels", 0.0)) for row in records], dtype=np.float32)
    has_cell = np.asarray([1.0 if str(row.get("has_cell", "")).lower() in {"true", "1", "yes"} else 0.0 for row in records], dtype=np.float32)
    return np.stack(
        [
            np.log1p(count),
            density,
            np.log1p(area),
            has_cell,
            np.sqrt(np.maximum(density, 0.0)),
        ],
        axis=1,
    ).astype(np.float32)


def process_image(root: str, image_id: str, base_variant: str, max_texture_dimension: int) -> dict:
    image_dir = Path(root) / PRET_DIR / image_id
    rgb = np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r")
    segments = np.load(image_dir / "superpixels.npy", mmap_mode="r")
    records = read_csv(image_dir / "superpixels.csv")
    num_segments = len(records)
    base = np.asarray(np.load(image_dir / f"tokens_{base_variant}.npy", mmap_mode="r"), dtype=np.float32)
    texture, texture_scale = _texture_block(np.asarray(rgb), np.asarray(segments), num_segments, max_texture_dimension)
    cellstats = _cellstats_block(records)
    token = weighted_concat([(base, 1.0), (texture, 0.75), (cellstats, 0.5)])
    token = safe_l2_normalize(token, axis=1)
    out_path = image_dir / f"tokens_{OUTPUT_VARIANT}.npy"
    np.save(out_path, token)
    report = {
        "image_id": image_id,
        "base_variant": base_variant,
        "output_variant": OUTPUT_VARIANT,
        "segments": num_segments,
        "base_dim": int(base.shape[1]),
        "texture_dim": int(texture.shape[1]),
        "texture_scale": float(texture_scale),
        "max_texture_dimension": int(max_texture_dimension),
        "cellstats_dim": int(cellstats.shape[1]),
        "output_dim": int(token.shape[1]),
        "finite": bool(np.isfinite(token).all()),
        "output_path": str(out_path),
    }
    write_json_atomic(image_dir / f"tokens_{OUTPUT_VARIANT}_validation.json", report)
    if not report["finite"]:
        raise RuntimeError(f"{image_id} enhanced token contains NaN/Inf")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Add texture/cell-stat enhanced PRET superpixel token variants.")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base_variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max_texture_dimension", type=int, default=8192)
    args = parser.parse_args()

    root = Path(args.source_root)
    image_ids = sorted(
        path.name
        for path in (root / PRET_DIR).iterdir()
        if path.is_dir() and (path / f"tokens_{args.base_variant}.npy").exists()
    )
    if args.image_id:
        requested = set(args.image_id)
        image_ids = [image_id for image_id in image_ids if image_id in requested]
    if not image_ids:
        raise RuntimeError("no images with base tokens found")

    reports = []
    if args.workers <= 1:
        for image_id in image_ids:
            reports.append(process_image(str(root), image_id, args.base_variant, args.max_texture_dimension))
            print(f"enhance_tokens {len(reports)}/{len(image_ids)} image_id={image_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_image, str(root), image_id, args.base_variant, args.max_texture_dimension): image_id
                for image_id in image_ids
            }
            for done, future in enumerate(as_completed(futures), start=1):
                report = future.result()
                reports.append(report)
                print(f"enhance_tokens {done}/{len(futures)} image_id={report['image_id']}", flush=True)
    print(json.dumps({"images": len(reports), "output_variant": OUTPUT_VARIANT, "passed": all(row["finite"] for row in reports)}, indent=2))


if __name__ == "__main__":
    main()
