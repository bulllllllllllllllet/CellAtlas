from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ.setdefault("VIPS_CONCURRENCY", "1")
import pyvips

# Each DataLoader worker visits many independent WSI files. Retaining libvips
# operation graphs across samples multiplies memory/resource use under DDP.
pyvips.cache_set_max(0)
pyvips.cache_set_max_mem(64 * 1024 * 1024)

Image.MAX_IMAGE_PIXELS = None

PAIR_COLUMNS = {"分割图路径": "gt_rel", "原图": "he_rel", "细胞核分类": "nuclei_class_rel", "实例分割结果": "nuclei_instance_rel"}


@dataclass(frozen=True)
class Pair:
    wsi_id: str
    patient_id: str
    he_path: Path
    gt_path: Path
    nuclei_class_path: Path
    nuclei_instance_path: Path


def read_pairs(config: dict[str, Any]) -> list[Pair]:
    csv_path, root = Path(config["data"]["pairs_csv"]), Path(config["data"]["common_root"])
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(PAIR_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pairs CSV missing required columns: {sorted(missing)}")
        pairs = []
        for row in reader:
            he = root / row["原图"]
            pairs.append(Pair(he.stem, he.stem, he, root / row["分割图路径"], root / row["细胞核分类"], root / row["实例分割结果"]))
    ids = [p.wsi_id for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate wsi_id in explicit pair CSV")
    return pairs


def rgb_counts(path: Path) -> Counter[tuple[int, int, int]]:
    """Count PNG colors after one decode; individual GT images exceed 300 MP."""
    try:
        import pyvips
    except ImportError as exc:
        raise ImportError("pyvips is required for bounded-memory GT auditing") from exc
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    if image.bands < 3:
        raise ValueError(f"GT must be RGB/RGBA, got {image.bands} bands: {path}")
    # Cropping a PNG through libvips replays decompression from the beginning.
    # Decode once, then bound the *analysis* working set by chunking pixels.
    raw = image.extract_band(0, n=3).write_to_memory()
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    result: Counter[tuple[int, int, int]] = Counter()
    chunk_pixels = 8_000_000
    for start in range(0, len(pixels), chunk_pixels):
        colors, counts = np.unique(pixels[start : start + chunk_pixels], axis=0, return_counts=True)
        result.update({tuple(map(int, color)): int(count) for color, count in zip(colors, counts, strict=True)})
    return result


def decode_gt(path: Path, class_map: list[dict[str, Any]], ignore_index: int) -> np.ndarray:
    if not class_map:
        raise ValueError("data.class_map is empty; run audit, confirm exact colors, then populate config")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    out = np.full(rgb.shape[:2], ignore_index, dtype=np.uint8)
    for item in class_map:
        color = np.asarray(item["rgb"], dtype=np.uint8)
        out[np.all(rgb == color, axis=-1)] = int(item["id"])
    return out


def decode_gt_patch(path: Path, x: int, y: int, width: int, height: int, class_map: list[dict[str, Any]], ignore_index: int) -> np.ndarray:
    """Decode one GT region without retaining a full-slide cache in workers."""
    image = pyvips.Image.new_from_file(str(path), access="random")
    if image.bands < 3:
        raise ValueError(f"GT must have at least three channels: {path}")
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError(f"patch outside GT bounds for {path}: {(x, y, width, height)}")
    crop = image.crop(x, y, width, height).extract_band(0, n=3)
    rgb = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(height, width, 3))
    out = np.full((height, width), ignore_index, dtype=np.uint8)
    for item in class_map:
        color = np.asarray(item["rgb"], dtype=np.uint8)
        out[np.all(rgb == color, axis=-1)] = int(item["id"])
    return out


def boundary_mask(mask: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = mask != ignore_index
    edge = np.zeros_like(valid)
    edge[1:] |= (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    edge[:, 1:] |= (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    return edge


def read_he_patch(path: Path, x_level0: int, y_level0: int, width_level0: int, height_level0: int, output_size: int) -> np.ndarray:
    """Read one original-resolution TIFF region and resize it to 10x."""
    image = pyvips.Image.new_from_file(str(path), access="random")
    if x_level0 < 0 or y_level0 < 0 or x_level0 + width_level0 > image.width or y_level0 + height_level0 > image.height:
        raise ValueError(f"patch outside HE bounds for {path}: {(x_level0, y_level0, width_level0, height_level0)}")
    patch = image.crop(x_level0, y_level0, width_level0, height_level0).resize(output_size / width_level0)
    if patch.bands < 3:
        raise ValueError(f"HE image has fewer than three channels: {path}")
    array = np.ndarray(buffer=patch.write_to_memory(), dtype=np.uint8, shape=(patch.height, patch.width, patch.bands))
    return array[..., :3].copy()
