from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import sobel

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import (
    NUM_CLASSES,
    PRET_DIR,
    majority_label,
    points_original_to_target,
    read_csv,
    safe_l2_normalize,
    weighted_concat,
    write_csv_atomic,
    write_json_atomic,
)


def read_he_at_target_scale(
    slide_path: str,
    target_shape: tuple[int, int],
    max_whole_read_mb: int,
    tile_size: int,
) -> tuple[np.ndarray, dict]:
    target_height, target_width = target_shape
    slide = openslide.OpenSlide(slide_path)
    try:
        original_width, original_height = slide.dimensions
        downsample = max(original_width / target_width, original_height / target_height)
        level = slide.get_best_level_for_downsample(downsample)
        level_width, level_height = slide.level_dimensions[level]
        estimated_mb = level_width * level_height * 3 / (1024 * 1024)
        if estimated_mb <= max_whole_read_mb:
            image = slide.read_region((0, 0), level, (level_width, level_height)).convert("RGB")
            image = image.resize((target_width, target_height), Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8), {
                "read_mode": "whole_level",
                "openslide_level": level,
                "level_dimensions": [level_width, level_height],
                "estimated_level_rgb_mb": estimated_mb,
            }

        output = np.empty((target_height, target_width, 3), dtype=np.uint8)
        scale_x = level_width / target_width
        scale_y = level_height / target_height
        level_downsample = float(slide.level_downsamples[level])
        for y0 in range(0, target_height, tile_size):
            for x0 in range(0, target_width, tile_size):
                x1 = min(target_width, x0 + tile_size)
                y1 = min(target_height, y0 + tile_size)
                lx0 = max(0, int(math.floor(x0 * scale_x)))
                ly0 = max(0, int(math.floor(y0 * scale_y)))
                lx1 = min(level_width, max(lx0 + 1, int(math.ceil(x1 * scale_x))))
                ly1 = min(level_height, max(ly0 + 1, int(math.ceil(y1 * scale_y))))
                location = (int(round(lx0 * level_downsample)), int(round(ly0 * level_downsample)))
                tile = slide.read_region(location, level, (lx1 - lx0, ly1 - ly0)).convert("RGB")
                tile = tile.resize((x1 - x0, y1 - y0), Image.BILINEAR)
                output[y0:y1, x0:x1] = np.asarray(tile, dtype=np.uint8)
        return output, {
            "read_mode": "tiled_level",
            "openslide_level": level,
            "level_dimensions": [level_width, level_height],
            "estimated_level_rgb_mb": estimated_mb,
            "tile_size_target": tile_size,
        }
    finally:
        slide.close()


def he_tissue_mask(rgb: np.ndarray, white_threshold: int = 240, saturation_threshold: int = 20) -> np.ndarray:
    image = rgb.astype(np.float32)
    max_channel = image.max(axis=2)
    min_channel = image.min(axis=2)
    saturation = max_channel - min_channel
    mean = image.mean(axis=2)
    return (mean < white_threshold) & (saturation > saturation_threshold)


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[str, int]]:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask & False, {"component_count": 0, "kept_component_count": 0, "largest_component_area": 0}
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    cleaned = keep[labels]
    return cleaned, {
        "component_count": int(count),
        "kept_component_count": int(keep.sum()),
        "largest_component_area": int(sizes[1:].max(initial=0)),
    }


def connectivity_distance_tissue_mask(
    rgb: np.ndarray,
    candidate_white_threshold: int,
    candidate_saturation_threshold: int,
    seed_white_threshold: int,
    seed_saturation_threshold: int,
    seed_min_area: int,
    distance_px: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    candidate = he_tissue_mask(rgb, candidate_white_threshold, candidate_saturation_threshold)
    seed_raw = he_tissue_mask(rgb, seed_white_threshold, seed_saturation_threshold)
    seed, seed_meta = remove_small_components(seed_raw, seed_min_area)
    distance = ndi.distance_transform_edt(~seed)
    tissue = candidate & (distance <= distance_px)
    return tissue, {
        "tissue_mask_mode": "connectivity_distance",
        "candidate_white_threshold": int(candidate_white_threshold),
        "candidate_saturation_threshold": int(candidate_saturation_threshold),
        "seed_white_threshold": int(seed_white_threshold),
        "seed_saturation_threshold": int(seed_saturation_threshold),
        "seed_min_area": int(seed_min_area),
        "distance_px": int(distance_px),
        "seed_component_count": seed_meta["component_count"],
        "seed_kept_component_count": seed_meta["kept_component_count"],
        "seed_largest_component_area": seed_meta["largest_component_area"],
        "candidate_fraction": float(candidate.mean()),
        "seed_fraction": float(seed.mean()),
        "tissue_fraction": float(tissue.mean()),
    }


def lowmag_loose_tissue_mask(
    rgb: np.ndarray,
    low_max_dim: int,
    min_component_area: int,
    close_iterations: int,
    dilate_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    height, width = rgb.shape[:2]
    if low_max_dim > 0 and max(height, width) > low_max_dim:
        scale = low_max_dim / max(height, width)
        low_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        low_rgb = np.asarray(Image.fromarray(rgb).resize(low_size, Image.BILINEAR), dtype=np.uint8)
    else:
        low_rgb = rgb
    image = low_rgb.astype(np.int16, copy=False)
    rgb_float = low_rgb.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb_float)
    mean_times3 = image.sum(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    strong_tissue = (mean_times3 < 244 * 3) & ((saturation >= 0.035) | (chroma >= 8))
    pale_tissue = (mean_times3 < 252 * 3) & (value <= 0.998) & ((saturation >= 0.010) | (chroma >= 2))
    od = -np.log(np.clip(rgb_float, 1 / 255, 1.0)).max(axis=2)
    edge = np.maximum(sobel(1.0 - value), sobel(od))
    edge_binary = edge >= float(np.percentile(edge, 88.0))
    edge_density = ndi.uniform_filter(edge_binary.astype(np.float32), size=7)
    edge_tissue = (edge_density >= 0.020) & (mean_times3 < 253 * 3)

    seed, seed_meta = remove_small_components(strong_tissue | edge_tissue, min_component_area)
    seed = ndi.binary_closing(seed, iterations=close_iterations)
    seed = ndi.binary_fill_holes(seed)
    loose = ndi.binary_dilation(seed, iterations=dilate_iterations) | pale_tissue
    loose, loose_meta = remove_small_components(loose, min_component_area)
    loose = ndi.binary_closing(loose, iterations=max(1, close_iterations // 2))
    loose = ndi.binary_fill_holes(loose)
    high_conf = ndi.binary_dilation(strong_tissue | edge_tissue, iterations=max(1, dilate_iterations // 2)) & loose
    low_conf = loose & ~high_conf

    if low_rgb.shape[:2] != (height, width):
        loose = np.asarray(Image.fromarray(loose.astype(np.uint8)).resize((width, height), Image.NEAREST), dtype=np.uint8) > 0
        high_conf = np.asarray(Image.fromarray(high_conf.astype(np.uint8)).resize((width, height), Image.NEAREST), dtype=np.uint8) > 0
        low_conf = np.asarray(Image.fromarray(low_conf.astype(np.uint8)).resize((width, height), Image.NEAREST), dtype=np.uint8) > 0
    return loose, high_conf, low_conf, {
        "tissue_mask_mode": "lowmag_loose",
        "lowmag_max_dim": int(low_max_dim),
        "lowmag_shape": list(low_rgb.shape[:2]),
        "lowmag_min_component_area": int(min_component_area),
        "lowmag_close_iterations": int(close_iterations),
        "lowmag_dilate_iterations": int(dilate_iterations),
        "lowmag_seed_component_count": seed_meta["component_count"],
        "lowmag_seed_kept_component_count": seed_meta["kept_component_count"],
        "lowmag_seed_largest_component_area": seed_meta["largest_component_area"],
        "lowmag_loose_component_count": loose_meta["component_count"],
        "lowmag_loose_kept_component_count": loose_meta["kept_component_count"],
        "lowmag_loose_largest_component_area": loose_meta["largest_component_area"],
        "tissue_fraction": float(loose.mean()),
        "high_conf_fraction": float(high_conf.mean()),
        "low_conf_fraction": float(low_conf.mean()),
        "low_conf_fraction_in_tissue": float(low_conf.sum() / max(1, loose.sum())),
    }


def slic_superpixels(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    target_sp_diameter: int,
    min_segments: int,
    max_segments: int,
    compactness: float,
    sigma: float,
    seed: int,
    convert2lab: bool,
) -> tuple[np.ndarray, dict]:
    from skimage.segmentation import slic

    tissue_area = int(np.sum(tissue_mask))
    requested = int(np.clip(tissue_area / max(1, target_sp_diameter**2), min_segments, max_segments))
    try:
        np.random.seed(seed)
        segments = slic(
            rgb,
            n_segments=requested,
            compactness=compactness,
            sigma=sigma,
            mask=tissue_mask,
            start_label=0,
            channel_axis=-1,
            convert2lab=convert2lab,
        ).astype(np.int32, copy=False)
        method = "slic"
    except Exception:
        segments = fallback_grid_segments(tissue_mask, target_sp_diameter)
        method = "fallback_grid"
    segments[~tissue_mask] = -1
    unique = np.unique(segments[segments >= 0])
    remapped = np.full_like(segments, -1, dtype=np.int32)
    valid = segments >= 0
    remapped[valid] = np.searchsorted(unique, segments[valid]).astype(np.int32, copy=False)
    return remapped, {
        "superpixel_method": method,
        "n_segments_requested": requested,
        "n_segments_actual": int(len(unique)),
        "tissue_area": tissue_area,
    }


def fallback_grid_segments(mask: np.ndarray, diameter: int) -> np.ndarray:
    height, width = mask.shape
    segments = np.full((height, width), -1, dtype=np.int32)
    segment_id = 0
    for y0 in range(0, height, diameter):
        for x0 in range(0, width, diameter):
            y1 = min(height, y0 + diameter)
            x1 = min(width, x0 + diameter)
            block = mask[y0:y1, x0:x1]
            if np.any(block):
                view = segments[y0:y1, x0:x1]
                view[block] = segment_id
                segment_id += 1
    return segments


def image_feature_blocks(rgb: np.ndarray, segments: np.ndarray, num_segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from skimage.color import rgb2lab

    rgb_float = rgb.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_float).astype(np.float32)
    height, width = segments.shape
    valid = segments >= 0
    segment_ids = segments[valid].reshape(-1)
    area = np.bincount(segment_ids, minlength=num_segments).astype(np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    sum_x = np.bincount(segment_ids, weights=xx[valid].reshape(-1), minlength=num_segments)
    sum_y = np.bincount(segment_ids, weights=yy[valid].reshape(-1), minlength=num_segments)
    centers = np.stack(
        [
            sum_x / np.maximum(area, 1),
            sum_y / np.maximum(area, 1),
        ],
        axis=1,
    ).astype(np.float32)

    feature_parts = []
    for image in (rgb_float, lab):
        pixels = image[valid].reshape(-1, 3)
        means = []
        stds = []
        for channel in range(3):
            values = pixels[:, channel]
            sums = np.bincount(segment_ids, weights=values, minlength=num_segments)
            sq_sums = np.bincount(segment_ids, weights=values * values, minlength=num_segments)
            mean = sums / np.maximum(area, 1)
            variance = np.maximum(sq_sums / np.maximum(area, 1) - mean * mean, 0)
            means.append(mean)
            stds.append(np.sqrt(variance))
        feature_parts.extend([np.stack(means, axis=1), np.stack(stds, axis=1)])
    image_rows = np.concatenate(feature_parts, axis=1).astype(np.float32)
    stats_rows = np.stack(
        [
            np.log1p(area),
            centers[:, 0] / max(1, width - 1),
            centers[:, 1] / max(1, height - 1),
        ],
        axis=1,
    ).astype(np.float32)
    return (
        image_rows,
        stats_rows,
        centers,
    )


def segment_gt_labels(segments: np.ndarray, gt_mask: np.ndarray, num_segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_segment = segments >= 0
    segment_ids = segments[valid_segment].reshape(-1)
    gt_values = np.asarray(gt_mask)[valid_segment].reshape(-1)
    valid_gt = (gt_values >= 0) & (gt_values < NUM_CLASSES)
    area = np.bincount(segment_ids, minlength=num_segments).astype(np.float64)
    valid_counts = np.bincount(segment_ids[valid_gt], minlength=num_segments).astype(np.float64)
    combined = segment_ids[valid_gt] * NUM_CLASSES + gt_values[valid_gt].astype(np.int64)
    counts = np.bincount(combined, minlength=num_segments * NUM_CLASSES).reshape(num_segments, NUM_CLASSES)
    labels = np.argmax(counts, axis=1).astype(np.int64)
    majority_counts = counts[np.arange(num_segments), labels].astype(np.float64)
    no_valid = valid_counts == 0
    labels[no_valid] = 255
    purity = np.divide(majority_counts, valid_counts, out=np.zeros(num_segments, dtype=np.float64), where=valid_counts > 0)
    valid_fraction = np.divide(valid_counts, area, out=np.zeros(num_segments, dtype=np.float64), where=area > 0)
    return labels, purity, valid_fraction


def load_cells_for_segments(output_root: Path, image_id: str, original_size: tuple[int, int], target_shape: tuple[int, int], segments: np.ndarray) -> dict:
    cell_dir = output_root / "cells" / image_id
    cells = read_csv(cell_dir / "cells.csv")
    labels = read_csv(cell_dir / "tissue_labels.csv")
    xy_original = np.asarray([[float(row["x_original"]), float(row["y_original"])] for row in cells], dtype=np.float64)
    xy_target = points_original_to_target(xy_original, original_size, target_shape)
    height, width = target_shape
    ix = np.clip(np.round(xy_target[:, 0]).astype(np.int64), 0, width - 1)
    iy = np.clip(np.round(xy_target[:, 1]).astype(np.int64), 0, height - 1)
    cell_segment = segments[iy, ix]
    valid = np.asarray([row["label_status"] == "valid" for row in labels]) & (cell_segment >= 0)
    return {
        "segment": cell_segment,
        "valid": valid,
        "raw": np.load(cell_dir / "raw.npy", mmap_mode="r"),
        "reg": np.load(cell_dir / "reg.npy", mmap_mode="r"),
        "proj": np.load(cell_dir / "proj.npy", mmap_mode="r"),
    }


def aggregate_cell_block(features: np.ndarray, cell_segment: np.ndarray, valid: np.ndarray, num_segments: int, segment_area: np.ndarray) -> np.ndarray:
    dim = int(features.shape[1])
    output = np.zeros((num_segments, dim * 4 + 3), dtype=np.float32)
    for segment_id in range(num_segments):
        indices = np.flatnonzero(valid & (cell_segment == segment_id))
        count = len(indices)
        if count:
            values = np.asarray(features[indices], dtype=np.float32)
            output[segment_id, :dim] = values.mean(axis=0)
            output[segment_id, dim: dim * 2] = values.std(axis=0)
            output[segment_id, dim * 2: dim * 3] = values.max(axis=0)
            output[segment_id, dim * 3: dim * 4] = np.median(values, axis=0)
        output[segment_id, dim * 4] = math.log1p(count)
        output[segment_id, dim * 4 + 1] = count / max(1.0, float(segment_area[segment_id]))
        output[segment_id, dim * 4 + 2] = 1.0 if count else 0.0
    return output


def nearest_patch_block(output_root: Path, image_id: str, original_size: tuple[int, int], centers_target: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    patch_dir = output_root / "patches" / image_id
    patches_path = patch_dir / "patches.csv"
    raw_path = patch_dir / "raw.npy"
    if not patches_path.is_file() or not raw_path.is_file():
        return np.zeros((len(centers_target), 768), dtype=np.float32), np.full(len(centers_target), -1, dtype=np.int64)
    patches = read_csv(patches_path)
    features = np.load(raw_path, mmap_mode="r")
    patch_xy_original = np.asarray(
        [[float(row["center_x_original"]), float(row["center_y_original"])] for row in patches],
        dtype=np.float64,
    )
    patch_xy_target = points_original_to_target(patch_xy_original, original_size, target_shape)
    _, nearest = cKDTree(patch_xy_target).query(centers_target, k=1, workers=-1)
    return np.asarray(features[nearest], dtype=np.float32), nearest.astype(np.int64)


def build_tokens(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image_stats = np.concatenate([blocks["image"], blocks["stats"]], axis=1)
    cell_reg = blocks["cell_reg"]
    patch = blocks["patch"]
    tokens = {
        "image_only": weighted_concat([(image_stats, 1.0)]),
        "cell_reg": weighted_concat([(cell_reg, 1.0)]),
        "patch_only": weighted_concat([(patch, 1.0)]),
        "image_cell_reg": weighted_concat([(image_stats, 1.0), (cell_reg, 1.0)]),
        "patch_cell_reg_equal_weight": weighted_concat([(patch, 1.0), (cell_reg, 1.0)]),
        "patch_cell_reg_low_patch_weight": weighted_concat([(patch, 0.5), (cell_reg, 1.0), (image_stats, 1.0)]),
        "image_patch_cell_reg": weighted_concat([(image_stats, 1.0), (patch, 1.0), (cell_reg, 1.0)]),
    }
    for cell_weight in (0.25, 0.5, 2.0):
        name = str(cell_weight).replace(".", "p")
        tokens[f"image_cell_reg_cellw{name}"] = weighted_concat(
            [(image_stats, 1.0), (cell_reg, cell_weight)]
        )
    return tokens


def process_slide(row: dict[str, str], args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    image_id = row["image_id"]
    slide_validation = json.loads((output_root / "cells" / image_id / "validation.json").read_text(encoding="utf-8"))
    original_size = tuple(int(value) for value in slide_validation["original_size"])
    gt_mask_full = np.load(output_root / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r")
    target_shape = tuple(int(value) for value in gt_mask_full.shape)
    smoke_downsample = 1.0
    if args.max_target_dimension and max(target_shape) > args.max_target_dimension:
        smoke_downsample = args.max_target_dimension / max(target_shape)
        target_shape = (
            max(1, int(round(target_shape[0] * smoke_downsample))),
            max(1, int(round(target_shape[1] * smoke_downsample))),
        )
    if args.target_sp_diameter == "auto_physical":
        effective_target_sp_diameter = max(
            1,
            int(round(args.base_sp_diameter * max(target_shape) / max(1, args.base_max_dimension))),
        )
    else:
        effective_target_sp_diameter = int(args.target_sp_diameter)
    rgb, read_report = read_he_at_target_scale(row["he_path"], target_shape, args.max_whole_read_mb, args.tile_size)
    if tuple(gt_mask_full.shape) == target_shape:
        gt_mask = np.asarray(gt_mask_full)
    else:
        gt_mask = np.asarray(
            Image.fromarray(np.asarray(gt_mask_full).astype(np.uint8)).resize(
                (target_shape[1], target_shape[0]), Image.NEAREST
            )
        )
    tissue_mask_mode = getattr(args, "tissue_mask_mode", "threshold")
    if tissue_mask_mode == "threshold":
        tissue = he_tissue_mask(rgb, args.white_threshold, args.saturation_threshold)
        tissue_report = {
            "tissue_mask_mode": "threshold",
            "candidate_white_threshold": int(args.white_threshold),
            "candidate_saturation_threshold": int(args.saturation_threshold),
            "tissue_fraction": float(tissue.mean()),
        }
    elif tissue_mask_mode == "connectivity_distance":
        tissue, tissue_report = connectivity_distance_tissue_mask(
            rgb,
            args.white_threshold,
            args.saturation_threshold,
            getattr(args, "seed_white_threshold", 234),
            getattr(args, "seed_saturation_threshold", 20),
            getattr(args, "seed_min_area", 50000),
            getattr(args, "distance_px", 256),
        )
    elif tissue_mask_mode == "lowmag_loose":
        tissue, high_conf_tissue, low_conf_tissue, tissue_report = lowmag_loose_tissue_mask(
            rgb,
            getattr(args, "lowmag_max_dim", 1200),
            getattr(args, "lowmag_min_component_area", 256),
            getattr(args, "lowmag_close_iterations", 5),
            getattr(args, "lowmag_dilate_iterations", 10),
        )
    else:
        raise ValueError(f"unsupported tissue_mask_mode: {tissue_mask_mode}")
    segments, slic_report = slic_superpixels(
        rgb,
        tissue,
        effective_target_sp_diameter,
        args.min_segments,
        args.max_segments,
        args.compactness,
        args.sigma,
        args.seed,
        args.slic_convert2lab,
    )
    num_segments = int(segments.max()) + 1 if np.any(segments >= 0) else 0
    if num_segments == 0:
        raise RuntimeError(f"{image_id} produced no HE superpixels")
    image_block, stats_block, centers = image_feature_blocks(rgb, segments, num_segments)
    segment_area = np.bincount(segments[segments >= 0].reshape(-1), minlength=num_segments).astype(np.float32)
    cell_data = load_cells_for_segments(output_root, image_id, original_size, target_shape, segments)
    cell_reg = aggregate_cell_block(cell_data["reg"], cell_data["segment"], cell_data["valid"], num_segments, segment_area)
    patch_block, patch_indices = nearest_patch_block(output_root, image_id, original_size, centers, target_shape)
    tokens = build_tokens({"image": image_block, "stats": stats_block, "cell_reg": cell_reg, "patch": patch_block})
    rng = np.random.default_rng(args.seed)
    tokens["random_token"] = safe_l2_normalize(
        rng.standard_normal(tokens["image_only"].shape).astype(np.float32),
        axis=1,
    )

    gt_labels, gt_purity, gt_valid_fraction = segment_gt_labels(segments, gt_mask, num_segments)
    cell_counts = np.bincount(
        cell_data["segment"][cell_data["valid"]].astype(np.int64),
        minlength=num_segments,
    )
    records = []
    for segment_id in range(num_segments):
        label = int(gt_labels[segment_id])
        purity = float(gt_purity[segment_id])
        valid_fraction = float(gt_valid_fraction[segment_id])
        count = int(cell_counts[segment_id])
        records.append(
            {
                "segment_id": segment_id,
                "center_x_10x": float(centers[segment_id, 0]),
                "center_y_10x": float(centers[segment_id, 1]),
                "area_10x_pixels": int(segment_area[segment_id]),
                "cell_count": count,
                "cell_density": float(count / max(1.0, segment_area[segment_id])),
                "has_cell": bool(count),
                "patch_index": int(patch_indices[segment_id]),
                "gt_tissue_label": label,
                "gt_label_purity": purity,
                "gt_valid_fraction": valid_fraction,
                "valid_for_retrieval": bool(valid_fraction >= args.min_gt_valid_fraction and purity >= args.min_gt_purity and 0 <= label < NUM_CLASSES),
            }
        )

    output_dir = output_root / PRET_DIR / image_id
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "he_10x_rgb.npy", rgb)
    np.save(output_dir / "he_tissue_mask.npy", tissue.astype(np.uint8))
    if tissue_mask_mode == "lowmag_loose":
        np.save(output_dir / "he_tissue_high_conf.npy", high_conf_tissue.astype(np.uint8))
        np.save(output_dir / "he_tissue_low_conf.npy", low_conf_tissue.astype(np.uint8))
    np.save(output_dir / "superpixels.npy", segments)
    for name, values in tokens.items():
        np.save(output_dir / f"tokens_{name}.npy", safe_l2_normalize(values, axis=1))
    write_csv_atomic(output_dir / "superpixels.csv", records)

    areas = segment_area[segment_area > 0]
    nan_count = int(sum(np.isnan(values).sum() for values in tokens.values()))
    inf_count = int(sum(np.isinf(values).sum() for values in tokens.values()))
    validation = {
        "passed": bool(num_segments > 0 and nan_count == 0 and inf_count == 0),
        "image_id": image_id,
        "he_10x_shape": list(rgb.shape),
        "gt_mask_shape": list(target_shape),
        "source_gt_mask_shape": list(gt_mask_full.shape),
        "original_size": list(original_size),
        "max_target_dimension": args.max_target_dimension,
        "smoke_downsample": smoke_downsample,
        "target_sp_diameter": args.target_sp_diameter,
        "effective_target_sp_diameter": effective_target_sp_diameter,
        "base_max_dimension": args.base_max_dimension,
        "base_sp_diameter": args.base_sp_diameter,
        "gt_used_for_superpixel": False,
        "slic_seed": args.seed,
        **tissue_report,
        "segment_area_min": int(areas.min(initial=0)),
        "segment_area_median": float(np.median(areas)) if len(areas) else 0.0,
        "segment_area_max": int(areas.max(initial=0)),
        "empty_cell_segment_ratio": float(np.mean([not row["has_cell"] for row in records])),
        "nan_token_count": nan_count,
        "inf_token_count": inf_count,
        "token_variants": sorted(tokens),
        **read_report,
        **slic_report,
    }
    write_json_atomic(output_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"{image_id} PRET token validation failed: {validation}")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PRET-style HE superpixel tokens.")
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv"))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--target_sp_diameter",
        default="64",
        help="Integer pixel diameter, or auto_physical to use base_sp_diameter * target_max_dim / base_max_dimension.",
    )
    parser.add_argument("--base_max_dimension", type=int, default=4096)
    parser.add_argument("--base_sp_diameter", type=int, default=64)
    parser.add_argument("--min_segments", type=int, default=1000)
    parser.add_argument("--max_segments", type=int, default=50000)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--slic_convert2lab", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--white_threshold", type=int, default=240)
    parser.add_argument("--saturation_threshold", type=int, default=20)
    parser.add_argument("--tissue_mask_mode", choices=["threshold", "connectivity_distance", "lowmag_loose"], default="threshold")
    parser.add_argument("--seed_white_threshold", type=int, default=234)
    parser.add_argument("--seed_saturation_threshold", type=int, default=20)
    parser.add_argument("--seed_min_area", type=int, default=50000)
    parser.add_argument("--distance_px", type=int, default=256)
    parser.add_argument("--lowmag_max_dim", type=int, default=1200)
    parser.add_argument("--lowmag_min_component_area", type=int, default=256)
    parser.add_argument("--lowmag_close_iterations", type=int, default=5)
    parser.add_argument("--lowmag_dilate_iterations", type=int, default=10)
    parser.add_argument("--min_gt_purity", type=float, default=0.7)
    parser.add_argument("--min_gt_valid_fraction", type=float, default=0.8)
    parser.add_argument("--max_whole_read_mb", type=int, default=2048)
    parser.add_argument("--tile_size", type=int, default=2048)
    parser.add_argument(
        "--max_target_dimension",
        type=int,
        default=0,
        help="Optional smoke-test downsampling cap for the 10x canvas; 0 keeps full 10x.",
    )
    args = parser.parse_args()
    rows = read_csv(args.manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    if args.workers <= 1:
        reports = [process_slide(row, args) for row in rows]
    else:
        reports = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_slide, row, args): row["image_id"] for row in rows}
            for completed, future in enumerate(as_completed(futures), start=1):
                report = future.result()
                reports.append(report)
                print(f"pret_superpixel_tokens {completed}/{len(futures)} image_id={report['image_id']}", flush=True)
    print(json.dumps({"slides": len(reports), "passed": all(row["passed"] for row in reports)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
