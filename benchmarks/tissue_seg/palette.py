from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


IGNORE_LABEL = 255


@dataclass(frozen=True)
class TissueClass:
    class_id: int
    name: str
    rgb: tuple[int, int, int]


# Approximate colors from the provided legend. Use inspect_gt_colors.py and a
# JSON palette to replace these with exact exported-label colors when needed.
DEFAULT_CLASSES: tuple[TissueClass, ...] = (
    TissueClass(0, "tumor_epithelium", (239, 45, 28)),
    TissueClass(1, "tumor_stroma", (248, 150, 30)),
    TissueClass(2, "background", (255, 255, 255)),
    TissueClass(3, "necrosis", (151, 84, 117)),
    TissueClass(4, "normal_gland", (255, 226, 50)),
    TissueClass(5, "normal_stroma", (72, 158, 203)),
    TissueClass(6, "submucosa_serosa", (170, 181, 84)),
    TissueClass(7, "muscle", (75, 153, 103)),
    TissueClass(8, "lymphocyte_aggregate", (69, 91, 122)),
    TissueClass(9, "mucus", (40, 157, 185)),
    TissueClass(10, "fat", (202, 208, 216)),
    TissueClass(11, "blood", (238, 48, 84)),
)


def load_palette(path: str | Path | None = None) -> list[TissueClass]:
    if path is None:
        return list(DEFAULT_CLASSES)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    classes: list[TissueClass] = []
    for item in raw["classes"]:
        classes.append(
            TissueClass(
                class_id=int(item["id"]),
                name=str(item["name"]),
                rgb=tuple(int(v) for v in item["rgb"]),
            )
        )
    return classes


def save_palette_template(path: str | Path) -> None:
    data = {
        "classes": [
            {"id": c.class_id, "name": c.name, "rgb": list(c.rgb)}
            for c in DEFAULT_CLASSES
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def color_gt_to_class_mask(
    gt_path: str | Path,
    palette: list[TissueClass],
    tolerance: float = 18.0,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    """Convert an RGB annotation PNG into a class-id mask using nearest color."""
    image = Image.open(gt_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]

    colors = np.asarray([c.rgb for c in palette], dtype=np.int32)
    class_ids = np.asarray([c.class_id for c in palette], dtype=np.uint8)
    lut = np.full(256 * 256 * 256, ignore_label, dtype=np.uint8)
    tol_sq = float(tolerance * tolerance)

    # Build a 24-bit RGB lookup table once. This is much faster than comparing
    # every pixel against every class color on very large annotation PNGs.
    chunk_size = 1_000_000
    for start in range(0, lut.size, chunk_size):
        end = min(start + chunk_size, lut.size)
        codes = np.arange(start, end, dtype=np.int32)
        code_rgb = np.stack(
            [
                (codes >> 16) & 255,
                (codes >> 8) & 255,
                codes & 255,
            ],
            axis=1,
        )
        diff = code_rgb[:, None, :] - colors[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist_sq, axis=1)
        min_dist_sq = dist_sq[np.arange(end - start), nearest]
        labels = class_ids[nearest]
        labels = labels.astype(np.uint8, copy=True)
        labels[min_dist_sq > tol_sq] = ignore_label
        lut[start:end] = labels

    flat = rgb.reshape(-1, 3)
    out = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        chunk = flat[start:end].astype(np.int32, copy=False)
        codes = (chunk[:, 0] << 16) + (chunk[:, 1] << 8) + chunk[:, 2]
        out[start:end] = lut[codes]
    return out.reshape(h, w)


def gt_to_class_mask(
    gt_path: str | Path,
    palette: list[TissueClass],
    tolerance: float = 18.0,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    """Load either a color-coded GT image or a numeric class-id mask."""
    image = Image.open(gt_path)
    if image.mode in {"RGB", "RGBA", "P"}:
        return color_gt_to_class_mask(gt_path, palette, tolerance, ignore_label)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return color_gt_to_class_mask(gt_path, palette, tolerance, ignore_label)

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported GT shape for {gt_path}: {arr.shape}")

    out = arr.astype(np.int64, copy=False)
    valid_class_ids = {c.class_id for c in palette}
    valid_values = np.isin(out, list(valid_class_ids))
    class_mask = np.full(out.shape, ignore_label, dtype=np.uint8)
    class_mask[valid_values] = out[valid_values].astype(np.uint8)
    return class_mask


def class_names(palette: list[TissueClass]) -> dict[int, str]:
    return {c.class_id: c.name for c in palette}
