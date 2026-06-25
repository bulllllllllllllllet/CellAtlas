from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.geometry import CoreTile, generate_core_tiles, map_point_to_gt
from new_inference_stream import inference as inference_base


Image.MAX_IMAGE_PIXELS = None
TOKEN_ORDER = "cellpose_label_ascending_reference"
CACHE_SCHEMA_VERSION = 2
CACHE_DIR_NAME = "tiles_label_order_v2"


@dataclass
class TileCells:
    labels: np.ndarray
    centers_xy: np.ndarray
    polygons: list[list[tuple[float, float]]]
    raw: np.ndarray
    reg: np.ndarray
    proj: np.ndarray


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _centers_and_labels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if labels.size == 0:
        return labels, np.empty((0, 2), dtype=np.float64)
    centers_yx = np.asarray(
        ndimage.center_of_mass(np.ones_like(mask, dtype=np.uint8), labels=mask, index=labels),
        dtype=np.float64,
    )
    return labels, centers_yx[:, ::-1]


def _polygon_for_label(
    mask: np.ndarray, label: int, bounds: tuple[slice, slice] | None
) -> list[tuple[float, float]]:
    if bounds is None:
        return []
    y_slice, x_slice = bounds
    y0, y1 = int(y_slice.start), int(y_slice.stop)
    x0, x1 = int(x_slice.start), int(x_slice.stop)
    binary = (mask[y0:y1, x0:x1] == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    return [(float(point[0][0] + x0), float(point[0][1] + y0)) for point in contour]


class FullResolutionEngine:
    def __init__(
        self,
        xcell_checkpoint: str,
        ctranspath_checkpoint: str,
        cellpose_checkpoint: str,
        device: str = "cuda",
        ctp_batch_size: int = 64,
        xcell_batch_size: int = 255,
    ) -> None:
        self.device = torch.device(device)
        self.ctp_batch_size = ctp_batch_size
        self.xcell_batch_size = xcell_batch_size
        self.cellpose = inference_base.load_cellpose_model(
            model_type=cellpose_checkpoint, device=self.device
        )
        self.ctranspath = inference_base.load_ctranspath(ctranspath_checkpoint, self.device)
        self.xcell = inference_base.load_xcell_model(xcell_checkpoint, self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_tile(self, tile_rgb: np.ndarray) -> TileCells:
        masks, _, _ = self.cellpose.eval(
            tile_rgb,
            diameter=18,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            do_3D=False,
        )
        masks = np.asarray(masks, dtype=np.int32)
        labels, centers_xy = _centers_and_labels(masks)
        if labels.size == 0:
            return TileCells(
                labels=labels,
                centers_xy=centers_xy,
                polygons=[],
                raw=np.empty((0, 768), dtype=np.float32),
                reg=np.empty((0, 64), dtype=np.float32),
                proj=np.empty((0, 64), dtype=np.float32),
            )

        raw_batches = []
        for start in range(0, len(labels), self.ctp_batch_size):
            crops: list[torch.Tensor] = []
            for label, (cx, cy) in zip(
                labels[start : start + self.ctp_batch_size],
                centers_xy[start : start + self.ctp_batch_size],
                strict=True,
            ):
                cy_int, cx_int = round(float(cy)), round(float(cx))
                crop = inference_base.crop_region(tile_rgb, cy_int, cx_int)
                crop_mask = inference_base.crop_region_mask(masks, cy_int, cx_int)
                attended = inference_base.apply_soft_attention(
                    crop, (crop_mask == int(label)).astype(np.float32)
                )
                crops.append(
                    self.preprocess(
                        Image.fromarray((np.clip(attended, 0, 1) * 255).astype(np.uint8))
                    )
                )
            batch = torch.stack(crops).to(self.device)
            with torch.inference_mode():
                raw_batches.append(self.ctranspath(batch).cpu().numpy())
        raw = np.concatenate(raw_batches, axis=0).astype(np.float32, copy=False)

        reg_batches: list[np.ndarray] = []
        proj_batches: list[np.ndarray] = []
        for start in range(0, len(raw), self.xcell_batch_size):
            values = raw[start : start + self.xcell_batch_size]
            n = len(values)
            padded = np.zeros((self.xcell_batch_size, raw.shape[1]), dtype=np.float32)
            valid_mask = np.zeros(self.xcell_batch_size, dtype=np.float32)
            padded[:n] = values
            valid_mask[:n] = 1
            with torch.inference_mode():
                _, reg, proj, _ = self.xcell(
                    raw_images=None,
                    x=torch.from_numpy(padded).unsqueeze(0).to(self.device),
                    mask=torch.from_numpy(valid_mask).unsqueeze(0).to(self.device),
                )
            reg_batches.append(reg[0, :n].cpu().numpy())
            proj_batches.append(proj[0, :n].cpu().numpy())
        reg = np.concatenate(reg_batches, axis=0).astype(np.float32, copy=False)
        proj = np.concatenate(proj_batches, axis=0).astype(np.float32, copy=False)

        if not (len(labels) == len(raw) == len(reg) == len(proj)):
            raise RuntimeError(
                f"tile cardinality mismatch labels={len(labels)} raw={len(raw)} "
                f"reg={len(reg)} proj={len(proj)}"
            )
        if not all(np.isfinite(array).all() for array in (raw, reg, proj)):
            raise RuntimeError("non-finite feature encountered")

        instance_bounds = ndimage.find_objects(masks)
        polygons = [
            _polygon_for_label(
                masks,
                int(label),
                instance_bounds[int(label) - 1] if int(label) <= len(instance_bounds) else None,
            )
            for label in labels
        ]
        return TileCells(labels, centers_xy, polygons, raw, reg, proj)


def _write_slide_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    raw: np.ndarray,
    reg: np.ndarray,
    proj: np.ndarray,
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("raw", raw), ("reg", reg), ("proj", proj)):
        path = output_dir / f"{name}.npy"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as file:
            np.save(file, value)
        os.replace(temporary, path)
    cells_path = output_dir / "cells.csv"
    temporary_cells = cells_path.with_suffix(cells_path.suffix + ".tmp")
    with open(temporary_cells, "w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "cell_index",
            "tile_index",
            "local_label",
            "x_original",
            "y_original",
            "x_gt",
            "y_gt",
            "core_boundary_distance",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    os.replace(temporary_cells, cells_path)
    polygons_path = output_dir / "polygons.jsonl"
    temporary_polygons = polygons_path.with_suffix(polygons_path.suffix + ".tmp")
    with open(temporary_polygons, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    {"cell_index": record["cell_index"], "polygon_original": record["polygon_original"]},
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary_polygons, polygons_path)
    validation_path = output_dir / "validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    with open(temporary_validation, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    os.replace(temporary_validation, validation_path)


def _cross_tile_near_duplicate_pairs(
    records: list[dict[str, Any]], radius: float = 2.0
) -> set[tuple[int, int]]:
    if len(records) < 2:
        return set()
    centers = np.asarray(
        [[record["x_original"], record["y_original"]] for record in records],
        dtype=np.float64,
    )
    from scipy.spatial import cKDTree

    return {
        (left, right)
        for left, right in cKDTree(centers).query_pairs(r=radius)
        if records[left]["tile_index"] != records[right]["tile_index"]
    }


def _deduplicate_cross_tile_records(
    records: list[dict[str, Any]],
    raw: np.ndarray,
    reg: np.ndarray,
    proj: np.ndarray,
    radius: float = 2.0,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    pairs = _cross_tile_near_duplicate_pairs(records, radius=radius)
    if not pairs:
        return records, raw, reg, proj, {
            "cells_before_dedup": len(records),
            "cross_tile_near_duplicates_before_dedup": 0,
            "deduplicated_cross_tile_cells": 0,
        }

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs:
        union(left, right)

    components: dict[int, list[int]] = {}
    for left, right in pairs:
        components.setdefault(find(left), []).extend([left, right])
    duplicate_components = [sorted(set(indices)) for indices in components.values()]

    keep = np.ones(len(records), dtype=bool)
    for indices in duplicate_components:
        winner = max(
            indices,
            key=lambda index: (
                float(records[index]["core_boundary_distance"]),
                -int(records[index]["tile_index"]),
                -int(records[index]["local_label"]),
                -index,
            ),
        )
        for index in indices:
            if index != winner:
                keep[index] = False

    kept_indices = np.flatnonzero(keep)
    deduped_records: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(kept_indices):
        record = dict(records[int(old_index)])
        record["cell_index"] = new_index
        deduped_records.append(record)

    stats = {
        "cells_before_dedup": len(records),
        "cross_tile_near_duplicates_before_dedup": len(pairs),
        "deduplicated_cross_tile_cells": int((~keep).sum()),
    }
    return deduped_records, raw[keep], reg[keep], proj[keep], stats


def _save_tile_cache(
    path: Path,
    tile: CoreTile,
    labels: np.ndarray,
    centers_global: np.ndarray,
    polygons_global: list[list[list[float]]],
    boundary_distance: np.ndarray,
    raw: np.ndarray,
    reg: np.ndarray,
    proj: np.ndarray,
) -> None:
    count = len(labels)
    if not (
        count
        == len(centers_global)
        == len(polygons_global)
        == len(boundary_distance)
        == len(raw)
        == len(reg)
        == len(proj)
    ):
        raise RuntimeError(f"tile {tile.tile_index} cache cardinality mismatch")
    if not all(np.isfinite(value).all() for value in (centers_global, boundary_distance, raw, reg, proj)):
        raise RuntimeError(f"tile {tile.tile_index} cache contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "wb") as file:
        np.savez(
            file,
            tile_index=np.asarray(tile.tile_index, dtype=np.int64),
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            token_order=np.asarray(TOKEN_ORDER),
            core_bounds=np.asarray(
                [tile.core_x0, tile.core_y0, tile.core_x1, tile.core_y1], dtype=np.int64
            ),
            read_bounds=np.asarray(
                [tile.read_x0, tile.read_y0, tile.read_x1, tile.read_y1], dtype=np.int64
            ),
            labels=np.asarray(labels, dtype=np.int64),
            centers_global=np.asarray(centers_global, dtype=np.float64),
            polygons_json=np.asarray(json.dumps(polygons_global, separators=(",", ":"))),
            boundary_distance=np.asarray(boundary_distance, dtype=np.float64),
            raw=np.asarray(raw, dtype=np.float32),
            reg=np.asarray(reg, dtype=np.float32),
            proj=np.asarray(proj, dtype=np.float32),
        )
    os.replace(temp_path, path)


def _load_tile_cache(path: Path, tile: CoreTile) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        tile_index = int(cached["tile_index"])
        cache_schema_version = int(cached["cache_schema_version"])
        token_order = str(cached["token_order"])
        core_bounds = cached["core_bounds"].tolist()
        read_bounds = cached["read_bounds"].tolist()
        expected_core = [tile.core_x0, tile.core_y0, tile.core_x1, tile.core_y1]
        expected_read = [tile.read_x0, tile.read_y0, tile.read_x1, tile.read_y1]
        if (
            tile_index != tile.tile_index
            or cache_schema_version != CACHE_SCHEMA_VERSION
            or token_order != TOKEN_ORDER
            or core_bounds != expected_core
            or read_bounds != expected_read
        ):
            raise RuntimeError(f"stale tile cache configuration: {path}")
        result = {
            "labels": np.asarray(cached["labels"], dtype=np.int64),
            "centers_global": np.asarray(cached["centers_global"], dtype=np.float64),
            "polygons_global": json.loads(str(cached["polygons_json"])),
            "boundary_distance": np.asarray(cached["boundary_distance"], dtype=np.float64),
            "raw": np.asarray(cached["raw"], dtype=np.float32),
            "reg": np.asarray(cached["reg"], dtype=np.float32),
            "proj": np.asarray(cached["proj"], dtype=np.float32),
        }
    count = len(result["labels"])
    if not all(
        len(result[key]) == count
        for key in (
            "centers_global",
            "polygons_global",
            "boundary_distance",
            "raw",
            "reg",
            "proj",
        )
    ):
        raise RuntimeError(f"corrupt tile cache cardinality: {path}")
    if not all(
        np.isfinite(result[key]).all()
        for key in ("centers_global", "boundary_distance", "raw", "reg", "proj")
    ):
        raise RuntimeError(f"corrupt tile cache values: {path}")
    return result


def process_slide(
    engine: FullResolutionEngine,
    row: dict[str, str],
    output_root: Path,
    tile_size: int,
    halo: int,
) -> dict[str, Any]:
    slide = openslide.OpenSlide(row["he_path"])
    width, height = slide.level_dimensions[0]
    with Image.open(row["tissue_gt_path"]) as gt_image:
        gt_size = gt_image.size
    tiles = generate_core_tiles(width, height, tile_size, halo)
    records: list[dict[str, Any]] = []
    raw_parts: list[np.ndarray] = []
    reg_parts: list[np.ndarray] = []
    proj_parts: list[np.ndarray] = []
    started = time.time()
    slide_output_dir = output_root / "cells" / row["image_id"]
    tile_cache_dir = slide_output_dir / CACHE_DIR_NAME
    cached_tiles = 0
    computed_tiles = 0

    try:
        for tile in tiles:
            cache_path = tile_cache_dir / f"tile_{tile.tile_index:04d}.npz"
            if cache_path.exists():
                cached = _load_tile_cache(cache_path, tile)
                cached_tiles += 1
                source = "cache"
            else:
                tile_rgb = np.asarray(
                    slide.read_region(
                        (tile.read_x0, tile.read_y0),
                        0,
                        (tile.read_x1 - tile.read_x0, tile.read_y1 - tile.read_y0),
                    ).convert("RGB")
                )
                tile_cells = engine.process_tile(tile_rgb)
                global_centers_all = tile_cells.centers_xy + np.asarray(
                    [tile.read_x0, tile.read_y0]
                )
                keep = np.asarray(
                    [tile.owns(float(x), float(y)) for x, y in global_centers_all], dtype=bool
                )
                global_centers = global_centers_all[keep]
                labels = tile_cells.labels[keep]
                polygons_global = [
                    [
                        [float(px + tile.read_x0), float(py + tile.read_y0)]
                        for px, py in tile_cells.polygons[index]
                    ]
                    for index in np.flatnonzero(keep)
                ]
                boundary_distance = np.asarray(
                    [
                        min(
                            x - tile.core_x0,
                            tile.core_x1 - x,
                            y - tile.core_y0,
                            tile.core_y1 - y,
                        )
                        for x, y in global_centers
                    ],
                    dtype=np.float64,
                )
                _save_tile_cache(
                    cache_path,
                    tile,
                    labels,
                    global_centers,
                    polygons_global,
                    boundary_distance,
                    tile_cells.raw[keep],
                    tile_cells.reg[keep],
                    tile_cells.proj[keep],
                )
                cached = _load_tile_cache(cache_path, tile)
                computed_tiles += 1
                source = "computed"

            for index, (x, y) in enumerate(cached["centers_global"]):
                x_gt, y_gt = map_point_to_gt(x, y, (width, height), gt_size)
                records.append(
                    {
                        "cell_index": len(records),
                        "tile_index": tile.tile_index,
                        "local_label": int(cached["labels"][index]),
                        "x_original": float(x),
                        "y_original": float(y),
                        "x_gt": float(x_gt),
                        "y_gt": float(y_gt),
                        "core_boundary_distance": float(cached["boundary_distance"][index]),
                        "polygon_original": cached["polygons_global"][index],
                    }
                )
            raw_parts.append(cached["raw"])
            reg_parts.append(cached["reg"])
            proj_parts.append(cached["proj"])
            print(
                f"[{row['image_id']}] tile={tile.tile_index + 1}/{len(tiles)} "
                f"kept={len(cached['labels'])} source={source}",
                flush=True,
            )
    finally:
        slide.close()

    raw = np.concatenate(raw_parts, axis=0) if raw_parts else np.empty((0, 768), np.float32)
    reg = np.concatenate(reg_parts, axis=0) if reg_parts else np.empty((0, 64), np.float32)
    proj = np.concatenate(proj_parts, axis=0) if proj_parts else np.empty((0, 64), np.float32)
    records, raw, reg, proj, dedup_stats = _deduplicate_cross_tile_records(records, raw, reg, proj)
    counts_equal = len(records) == len(raw) == len(reg) == len(proj)
    cross_tile_near_duplicates = len(_cross_tile_near_duplicate_pairs(records, radius=2.0))
    report = {
        "image_id": row["image_id"],
        "original_size": [width, height],
        "gt_size": list(gt_size),
        "tile_size": tile_size,
        "halo": halo,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "tile_cache_directory": CACHE_DIR_NAME,
        "xcell_token_order": TOKEN_ORDER,
        "tiles": len(tiles),
        "tiles_cached": cached_tiles,
        "tiles_computed": computed_tiles,
        **dedup_stats,
        "cells": len(records),
        "raw_shape": list(raw.shape),
        "reg_shape": list(reg.shape),
        "proj_shape": list(proj.shape),
        "counts_equal": counts_equal,
        "cross_tile_near_duplicates_within_2px": cross_tile_near_duplicates,
        "features_finite": all(np.isfinite(value).all() for value in (raw, reg, proj)),
        "elapsed_seconds": time.time() - started,
    }
    report["passed"] = (
        report["counts_equal"]
        and report["features_finite"]
        and report["cross_tile_near_duplicates_within_2px"] == 0
        and len(records) > 0
    )
    if not report["passed"]:
        raise RuntimeError(f"slide validation failed: {report}")
    _write_slide_outputs(slide_output_dir, records, raw, reg, proj, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact-checkpoint full-resolution GDPH inference.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument(
        "--xcell_checkpoint",
        default="experiments/crc012_holdout3/20260515_160711/he_model_best.pth",
    )
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--cellpose_checkpoint", default="/home/zyh/.cellpose/models/cpsam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile_size", type=int, default=4096)
    parser.add_argument("--halo", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = _read_csv(args.manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
        missing = requested - {row["image_id"] for row in rows}
        if missing:
            raise ValueError(f"image ids not found in manifest: {sorted(missing)}")
    if not args.force:
        incomplete_rows = []
        for row in rows:
            validation_path = Path(args.output_root) / "cells" / row["image_id"] / "validation.json"
            if validation_path.exists():
                with open(validation_path, "r", encoding="utf-8") as file:
                    validation = json.load(file)
                if (
                    validation.get("passed")
                    and validation.get("tile_size") == args.tile_size
                    and validation.get("halo") == args.halo
                    and validation.get("features_finite") is True
                    and validation.get("cross_tile_near_duplicates_within_2px") == 0
                    and validation.get("cache_schema_version") == CACHE_SCHEMA_VERSION
                    and validation.get("xcell_token_order") == TOKEN_ORDER
                ):
                    print(f"[{row['image_id']}] already complete", flush=True)
                    continue
            incomplete_rows.append(row)
        rows = incomplete_rows
    if not rows:
        print("all requested slides are already complete", flush=True)
        return
    engine = FullResolutionEngine(
        args.xcell_checkpoint,
        args.ctranspath_checkpoint,
        args.cellpose_checkpoint,
        args.device,
    )
    for row in rows:
        report = process_slide(
            engine, row, Path(args.output_root), tile_size=args.tile_size, halo=args.halo
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
