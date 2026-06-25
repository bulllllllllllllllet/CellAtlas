from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.geometry import map_point_to_gt
from benchmarks.tissue_seg.palette import IGNORE_LABEL, gt_to_class_mask, load_palette


Image.MAX_IMAGE_PIXELS = None


def polygon_majority_label(
    polygon_original: list[list[float]],
    centroid_original: tuple[float, float],
    original_size: tuple[int, int],
    gt_mask: np.ndarray,
    num_classes: int,
) -> tuple[int, float, int]:
    gt_height, gt_width = gt_mask.shape
    centroid_gt = map_point_to_gt(
        centroid_original[0], centroid_original[1], original_size, (gt_width, gt_height)
    )
    if len(polygon_original) < 3:
        x = int(np.clip(round(centroid_gt[0]), 0, gt_width - 1))
        y = int(np.clip(round(centroid_gt[1]), 0, gt_height - 1))
        label = int(gt_mask[y, x])
        return label, 1.0 if 0 <= label < num_classes else 0.0, 1

    polygon_gt = np.asarray(
        [map_point_to_gt(x, y, original_size, (gt_width, gt_height)) for x, y in polygon_original],
        dtype=np.float64,
    )
    x0 = max(0, int(np.floor(polygon_gt[:, 0].min())))
    y0 = max(0, int(np.floor(polygon_gt[:, 1].min())))
    x1 = min(gt_width, int(np.ceil(polygon_gt[:, 0].max())) + 1)
    y1 = min(gt_height, int(np.ceil(polygon_gt[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return IGNORE_LABEL, 0.0, 0

    local_polygon = polygon_gt - np.asarray([x0, y0])
    coverage_image = Image.new("1", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(coverage_image).polygon(
        [(float(x), float(y)) for x, y in local_polygon], fill=1
    )
    coverage = np.asarray(coverage_image, dtype=np.uint8)
    values = np.asarray(gt_mask[y0:y1, x0:x1])[coverage > 0]
    valid = values[(values >= 0) & (values < num_classes)]
    if valid.size == 0:
        return IGNORE_LABEL, 0.0, int(values.size)
    counts = np.bincount(valid.astype(np.int64), minlength=num_classes)
    label = int(np.argmax(counts))
    # Ignore/unmapped GT pixels must reduce confidence.  Dividing only by the
    # valid subset would incorrectly turn a tiny valid overlap into purity 1.
    return label, float(counts[label] / values.size), int(values.size)


def label_status(label: int, purity: float) -> str:
    if label == IGNORE_LABEL or purity < 0.5:
        return "ignore"
    if purity < 0.7:
        return "boundary"
    return "valid"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def label_slide(row: dict[str, str], output_root: Path) -> dict:
    slide_dir = output_root / "cells" / row["image_id"]
    cells = _read_csv(slide_dir / "cells.csv")
    polygons: dict[int, list[list[float]]] = {}
    with open(slide_dir / "polygons.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            polygons[int(item["cell_index"])] = item["polygon_original"]
    cell_indices = [int(cell["cell_index"]) for cell in cells]
    polygon_indices = set(polygons)
    indices_equal = (
        len(cell_indices) == len(set(cell_indices))
        and set(cell_indices) == polygon_indices
    )
    if not indices_equal:
        raise RuntimeError(
            f"cell/polygon index mismatch: cells={len(cell_indices)} "
            f"unique_cells={len(set(cell_indices))} polygons={len(polygon_indices)}"
        )

    with Image.open(row["he_path"]) as image:
        original_size = image.size
    palette = load_palette(output_root / "manifests" / "gdph_tissue_palette.json")
    gt_cache = output_root / "masks" / f"{row['image_id']}_gt_mask.npy"
    if gt_cache.exists():
        gt_mask = np.load(gt_cache, mmap_mode="r")
    else:
        gt_mask = gt_to_class_mask(row["tissue_gt_path"], palette, tolerance=35.0)
        temporary = gt_cache.with_suffix(gt_cache.suffix + ".tmp")
        with open(temporary, "wb") as file:
            np.save(file, gt_mask)
        temporary.replace(gt_cache)

    output_rows = []
    for cell in cells:
        cell_index = int(cell["cell_index"])
        label, purity, pixels = polygon_majority_label(
            polygons[cell_index],
            (float(cell["x_original"]), float(cell["y_original"])),
            original_size,
            gt_mask,
            len(palette),
        )
        output_rows.append(
            {
                "cell_index": cell_index,
                "gt_tissue_label": label,
                "gt_label_purity": purity,
                "gt_polygon_pixels": pixels,
                "label_status": label_status(label, purity),
            }
        )

    labels_path = slide_dir / "tissue_labels.csv"
    temporary_labels = labels_path.with_suffix(labels_path.suffix + ".tmp")
    with open(temporary_labels, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_labels.replace(labels_path)
    status_counts = {
        status: sum(item["label_status"] == status for item in output_rows)
        for status in ("valid", "boundary", "ignore")
    }
    purities = np.asarray([item["gt_label_purity"] for item in output_rows])
    gt_height, gt_width = gt_mask.shape
    report = {
        "image_id": row["image_id"],
        "cells": len(cells),
        "labels": len(output_rows),
        "counts_equal": len(cells) == len(output_rows),
        "indices_equal": indices_equal,
        "original_size_wh": list(original_size),
        "gt_size_wh": [int(gt_width), int(gt_height)],
        "original_to_gt_scale_xy": [
            gt_width / original_size[0],
            gt_height / original_size[1],
        ],
        "purity_definition": "majority_class_pixels / all_polygon_pixels",
        "purity_thresholds": {"valid": 0.7, "boundary_min": 0.5},
        "purity_min": float(purities.min()),
        "purity_median": float(np.median(purities)),
        "purity_max": float(purities.max()),
        "status_counts": status_counts,
        "passed": bool(
            len(cells) == len(output_rows)
            and len(cells) > 0
            and indices_equal
            and sum(status_counts.values()) == len(cells)
            and np.isfinite(purities).all()
            and (purities >= 0).all()
            and (purities <= 1).all()
            and status_counts["valid"] > 0
        ),
    }
    validation_path = slide_dir / "label_validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    with open(temporary_validation, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary_validation.replace(validation_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign polygon-majority GDPH tissue labels.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image_id", action="append", default=[])
    args = parser.parse_args()
    rows = _read_csv(Path(args.manifest))
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    for row in rows:
        report = label_slide(row, Path(args.output_root))
        print(json.dumps(report, ensure_ascii=False))
        if not report["passed"]:
            raise SystemExit(f"label validation failed for {row['image_id']}")


if __name__ == "__main__":
    main()
