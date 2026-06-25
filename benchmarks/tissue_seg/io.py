from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.tissue_seg.palette import IGNORE_LABEL, TissueClass, gt_to_class_mask


@dataclass(frozen=True)
class MappingRow:
    image_id: str
    gt_path: Path
    masks_path: Path
    reg_path: Path | None
    proj_path: Path | None
    raw_path: Path | None
    he_path: Path | None


def read_mapping_csv(path: str | Path) -> list[MappingRow]:
    rows: list[MappingRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "gt_path", "masks_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"mapping CSV missing columns: {sorted(missing)}")

        for raw in reader:
            rows.append(
                MappingRow(
                    image_id=raw["image_id"],
                    gt_path=Path(raw["gt_path"]),
                    masks_path=Path(raw["masks_path"]),
                    reg_path=_optional_path(raw.get("reg_path")),
                    proj_path=_optional_path(raw.get("proj_path")),
                    raw_path=_optional_path(raw.get("raw_path")),
                    he_path=_optional_path(raw.get("he_path")),
                )
            )
    return rows


def _optional_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return Path(value)


def load_feature(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D feature array, got {arr.shape}")
    return np.asarray(arr)


def load_or_create_gt_mask(
    gt_path: Path,
    output_path: Path,
    palette: list[TissueClass],
    tolerance: float,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path, mmap_mode="r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = gt_to_class_mask(gt_path, palette, tolerance=tolerance)
    np.save(output_path, mask)
    return mask


def cell_centroids_and_gt_labels(
    masks: np.ndarray,
    gt_mask: np.ndarray,
    ignore_label: int = IGNORE_LABEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell ids, integer centroids (x, y), and GT label sampled at centroid."""
    masks = np.asarray(masks)
    gt_mask = np.asarray(gt_mask)
    if masks.shape != gt_mask.shape:
        raise ValueError(f"masks shape {masks.shape} != gt shape {gt_mask.shape}")

    labels = np.unique(masks)
    labels = labels[labels > 0]
    if labels.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.uint8),
        )

    ys, xs = np.nonzero(masks)
    point_labels = masks[ys, xs].astype(np.int64, copy=False)
    max_label = int(point_labels.max())
    counts = np.bincount(point_labels, minlength=max_label + 1)
    sum_y = np.bincount(point_labels, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(point_labels, weights=xs, minlength=max_label + 1)
    centroid_x = np.rint(sum_x[labels] / counts[labels]).astype(np.int64)
    centroid_y = np.rint(sum_y[labels] / counts[labels]).astype(np.int64)
    centroids = np.stack([centroid_x, centroid_y], axis=1)
    h, w = gt_mask.shape
    centroids[:, 0] = np.clip(centroids[:, 0], 0, w - 1)
    centroids[:, 1] = np.clip(centroids[:, 1], 0, h - 1)
    gt_labels = gt_mask[centroids[:, 1], centroids[:, 0]].astype(np.uint8)
    gt_labels = np.where(gt_labels == ignore_label, ignore_label, gt_labels).astype(np.uint8)
    return labels.astype(np.int64), centroids, gt_labels
