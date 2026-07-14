# 作用：为 v3 阶段 A 构建 small/medium/large 多尺度 superpixel 与 region token 标准输出。

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2 import pret_superpixel_tokens
from benchmarks.gdph_v2.pret_aaai_enhance_tokens import OUTPUT_VARIANT, process_image
from benchmarks.gdph_v2.pret_utils import NUM_CLASSES, read_csv


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
DATA_MANIFEST = V3_ROOT / "pret_superpixel" / "data_manifest_v3.csv"
LEGACY_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2")
REPORT_DIR = Path(__file__).resolve().parent

SCALES = {
    "small": 0.5,
    "medium": 1.0,
    "large": 2.0,
}

STANDARD_FIELDS = [
    "segment_id",
    "area",
    "center_x",
    "center_y",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "cell_count",
    "cell_density",
    "gt_majority_label",
    "gt_target_fraction",
    "gt_purity",
    "valid_fraction",
]


@dataclass(frozen=True)
class ScaleJob:
    scale: str
    factor: float
    root: Path
    base_sp_diameter: int


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ensure_link(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() and Path(os.readlink(link_path)) == target_path:
        return
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def ensure_generation_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("cells", "masks", "patches"):
        ensure_link(root / name, LEGACY_ROOT / name)


def process_generation_slide(row: dict[str, str], job: ScaleJob, args: argparse.Namespace) -> dict[str, object]:
    generation_row = dict(row)
    generation_row["tissue_gt_path"] = row["gt_mask_path"]
    namespace = SimpleNamespace(
        output_root=str(job.root),
        target_sp_diameter="auto_physical",
        base_max_dimension=args.base_max_dimension,
        base_sp_diameter=job.base_sp_diameter,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        seed=args.seed,
        slic_convert2lab=args.slic_convert2lab,
        white_threshold=args.white_threshold,
        saturation_threshold=args.saturation_threshold,
        tissue_mask_mode=args.tissue_mask_mode,
        seed_white_threshold=args.seed_white_threshold,
        seed_saturation_threshold=args.seed_saturation_threshold,
        seed_min_area=args.seed_min_area,
        distance_px=args.distance_px,
        lowmag_max_dim=args.lowmag_max_dim,
        lowmag_min_component_area=args.lowmag_min_component_area,
        lowmag_close_iterations=args.lowmag_close_iterations,
        lowmag_dilate_iterations=args.lowmag_dilate_iterations,
        min_gt_purity=args.min_gt_purity,
        min_gt_valid_fraction=args.min_gt_valid_fraction,
        max_whole_read_mb=args.max_whole_read_mb,
        tile_size=args.tile_size,
        max_target_dimension=args.max_target_dimension,
    )
    report = pret_superpixel_tokens.process_slide(generation_row, namespace)
    enhance = process_image(str(job.root), row["image_id"], "image_cell_reg_cellw0p5", args.max_texture_dimension)
    report["scale"] = job.scale
    report["enhanced_output_dim"] = enhance["output_dim"]
    return report


def generate_scale(job: ScaleJob, rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    ensure_generation_root(job.root)
    pending = [
        row
        for row in rows
        if args.force
        or not (job.root / "pret_superpixel" / row["image_id"] / f"tokens_{OUTPUT_VARIANT}.npy").exists()
    ]
    reports: list[dict[str, object]] = []
    if not pending:
        for row in rows:
            validation_path = job.root / "pret_superpixel" / row["image_id"] / "validation.json"
            report = json.loads(validation_path.read_text(encoding="utf-8"))
            report["scale"] = job.scale
            reports.append(report)
        return reports

    if args.workers <= 1:
        for index, row in enumerate(pending, start=1):
            report = process_generation_slide(row, job, args)
            reports.append(report)
            print(f"generated {job.scale} {index}/{len(pending)} image_id={row['image_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_generation_slide, row, job, args): row["image_id"]
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                report = future.result()
                reports.append(report)
                print(f"generated {job.scale} {index}/{len(futures)} image_id={report['image_id']}", flush=True)
    return reports


def bbox_rows(segments: np.ndarray, records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_segments = len(records)
    x0 = np.full(num_segments, -1, dtype=np.int64)
    y0 = np.full(num_segments, -1, dtype=np.int64)
    x1 = np.full(num_segments, -1, dtype=np.int64)
    y1 = np.full(num_segments, -1, dtype=np.int64)
    width = segments.shape[1]
    x0.fill(width)
    y0.fill(segments.shape[0])
    x_coords = np.arange(width, dtype=np.int64)
    for chunk_y0 in range(0, segments.shape[0], 512):
        chunk_y1 = min(segments.shape[0], chunk_y0 + 512)
        block = np.asarray(segments[chunk_y0:chunk_y1])
        valid = block >= 0
        if not np.any(valid):
            continue
        segment_ids = block[valid].astype(np.int64, copy=False)
        yy = np.broadcast_to(np.arange(chunk_y0, chunk_y1, dtype=np.int64)[:, None], block.shape)[valid]
        xx = np.broadcast_to(x_coords[None, :], block.shape)[valid]
        np.minimum.at(x0, segment_ids, xx)
        np.minimum.at(y0, segment_ids, yy)
        np.maximum.at(x1, segment_ids, xx + 1)
        np.maximum.at(y1, segment_ids, yy + 1)
    x0[x0 == width] = -1
    y0[y0 == segments.shape[0]] = -1
    return x0, y0, x1, y1


def adjacency_edges(segments: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for left, right in (
        (segments[:, :-1], segments[:, 1:]),
        (segments[:-1, :], segments[1:, :]),
    ):
        mask = (left >= 0) & (right >= 0) & (left != right)
        if not np.any(mask):
            continue
        pairs = np.stack([left[mask], right[mask]], axis=1).astype(np.int64, copy=False)
        pairs.sort(axis=1)
        edges.update(map(tuple, np.unique(pairs, axis=0).tolist()))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def standardize_superpixels(records: list[dict[str, str]], segments: np.ndarray) -> list[dict[str, object]]:
    x0, y0, x1, y1 = bbox_rows(segments, records)
    output: list[dict[str, object]] = []
    for index, row in enumerate(records):
        purity = float(row.get("gt_label_purity", row.get("gt_purity", 0.0)))
        output.append(
            {
                "segment_id": int(row["segment_id"]),
                "area": int(float(row.get("area_10x_pixels", row.get("area", 0)))),
                "center_x": float(row.get("center_x_10x", row.get("center_x", 0.0))),
                "center_y": float(row.get("center_y_10x", row.get("center_y", 0.0))),
                "bbox_x0": int(x0[index]),
                "bbox_y0": int(y0[index]),
                "bbox_x1": int(x1[index]),
                "bbox_y1": int(y1[index]),
                "cell_count": int(float(row.get("cell_count", 0))),
                "cell_density": float(row.get("cell_density", 0.0)),
                "gt_majority_label": int(float(row.get("gt_tissue_label", row.get("gt_majority_label", 255)))),
                "gt_target_fraction": purity,
                "gt_purity": purity,
                "valid_fraction": float(row.get("gt_valid_fraction", row.get("valid_fraction", 0.0))),
            }
        )
    return output


def gt_label_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["gt_majority_label"])].append(row)
    output: list[dict[str, object]] = []
    for label in sorted(by_label):
        group = by_label[label]
        valid_group = [row for row in group if int(row["gt_majority_label"]) < NUM_CLASSES and float(row["valid_fraction"]) > 0]
        output.append(
            {
                "gt_majority_label": label,
                "segment_count": len(group),
                "valid_segment_count": len(valid_group),
                "area": int(sum(int(row["area"]) for row in group)),
                "mean_gt_purity": float(np.mean([float(row["gt_purity"]) for row in group])) if group else 0.0,
                "mean_valid_fraction": float(np.mean([float(row["valid_fraction"]) for row in group])) if group else 0.0,
            }
        )
    return output


def load_medium_rows(source_dir: Path) -> list[dict[str, object]]:
    records = read_csv(source_dir / "superpixels.csv")
    segments = np.load(source_dir / "superpixels.npy", mmap_mode="r")
    return standardize_superpixels(records, np.asarray(segments))


def normalize_token(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def write_small_from_medium(source_dir: Path, dest_dir: Path, chunk_rows: int) -> None:
    medium = np.load(source_dir / "superpixels.npy", mmap_mode="r")
    medium_rows = load_medium_rows(source_dir)
    num_medium = len(medium_rows)
    x_mid = np.asarray([(row["bbox_x0"] + row["bbox_x1"]) / 2.0 for row in medium_rows], dtype=np.float32)
    y_mid = np.asarray([(row["bbox_y0"] + row["bbox_y1"]) / 2.0 for row in medium_rows], dtype=np.float32)

    raw_present = np.zeros(num_medium * 4, dtype=bool)
    raw_path = dest_dir / "_small_raw_ids.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.int32, shape=medium.shape)
    raw[:] = -1
    height, width = medium.shape
    x_coords = np.arange(width, dtype=np.float32)
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        block = np.asarray(medium[y0:y1])
        valid = block >= 0
        if not np.any(valid):
            continue
        sid = block[valid].astype(np.int64, copy=False)
        yy = np.broadcast_to(np.arange(y0, y1, dtype=np.float32)[:, None], block.shape)[valid]
        xx = np.broadcast_to(x_coords[None, :], block.shape)[valid]
        quadrant = (xx >= x_mid[sid]).astype(np.int64) + 2 * (yy >= y_mid[sid]).astype(np.int64)
        values = sid * 4 + quadrant
        out = raw[y0:y1]
        out[valid] = values.astype(np.int32, copy=False)
        raw_present[np.unique(values)] = True
    raw.flush()

    mapping = np.full(num_medium * 4, -1, dtype=np.int32)
    present_ids = np.flatnonzero(raw_present)
    mapping[present_ids] = np.arange(len(present_ids), dtype=np.int32)
    segments = np.lib.format.open_memmap(dest_dir / "superpixels.npy", mode="w+", dtype=np.int32, shape=medium.shape)
    segments[:] = -1
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        block = np.asarray(raw[y0:y1])
        valid = block >= 0
        out = segments[y0:y1]
        out[valid] = mapping[block[valid]]
    segments.flush()
    raw_path.unlink(missing_ok=True)

    parent = (present_ids // 4).astype(np.int64)
    write_derived_tokens(source_dir, dest_dir, parent, mode="copy")
    write_derived_records(dest_dir, medium_rows, parent, mode="split")


def write_large_from_medium(source_dir: Path, dest_dir: Path, factor: float, chunk_rows: int) -> None:
    medium = np.load(source_dir / "superpixels.npy", mmap_mode="r")
    medium_rows = load_medium_rows(source_dir)
    diameter = float(np.sqrt(np.median([max(1, row["area"]) for row in medium_rows])) * factor)
    center_x = np.asarray([row["center_x"] for row in medium_rows], dtype=np.float64)
    center_y = np.asarray([row["center_y"] for row in medium_rows], dtype=np.float64)
    grid_x = np.floor(center_x / max(1.0, diameter)).astype(np.int64)
    grid_y = np.floor(center_y / max(1.0, diameter)).astype(np.int64)
    keys = list(zip(grid_x.tolist(), grid_y.tolist(), strict=True))
    key_to_group: dict[tuple[int, int], int] = {}
    parent_to_group = np.empty(len(medium_rows), dtype=np.int32)
    for parent_id, key in enumerate(keys):
        if key not in key_to_group:
            key_to_group[key] = len(key_to_group)
        parent_to_group[parent_id] = key_to_group[key]

    segments = np.lib.format.open_memmap(dest_dir / "superpixels.npy", mode="w+", dtype=np.int32, shape=medium.shape)
    segments[:] = -1
    height = medium.shape[0]
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        block = np.asarray(medium[y0:y1])
        valid = block >= 0
        out = segments[y0:y1]
        out[valid] = parent_to_group[block[valid]]
    segments.flush()

    groups: list[list[int]] = [[] for _ in range(len(key_to_group))]
    for parent_id, group_id in enumerate(parent_to_group.tolist()):
        groups[group_id].append(parent_id)
    representative_parent = np.asarray([members[0] for members in groups], dtype=np.int64)
    write_derived_tokens(source_dir, dest_dir, representative_parent, mode="aggregate", groups=groups, medium_rows=medium_rows)
    write_derived_records(dest_dir, medium_rows, representative_parent, mode="merge", groups=groups)


def write_derived_tokens(
    source_dir: Path,
    dest_dir: Path,
    parent: np.ndarray,
    mode: str,
    groups: list[list[int]] | None = None,
    medium_rows: list[dict[str, object]] | None = None,
) -> None:
    for filename in ("tokens_image_cell_reg_cellw0p5.npy", "tokens_image_cell_reg_texture_cellstats.npy"):
        token = np.asarray(np.load(source_dir / filename, mmap_mode="r"), dtype=np.float32)
        if mode == "copy":
            derived = token[parent]
        else:
            assert groups is not None and medium_rows is not None
            derived = np.zeros((len(groups), token.shape[1]), dtype=np.float32)
            areas = np.asarray([float(row["area"]) for row in medium_rows], dtype=np.float32)
            for group_id, members in enumerate(groups):
                weights = areas[members]
                derived[group_id] = np.average(token[members], axis=0, weights=np.maximum(weights, 1.0))
            derived = normalize_token(derived).astype(np.float32)
        np.save(dest_dir / filename, derived)
    ensure_link(dest_dir / "tokens_image_cell_reg.npy", dest_dir / "tokens_image_cell_reg_cellw0p5.npy")


def write_derived_records(
    dest_dir: Path,
    medium_rows: list[dict[str, object]],
    parent: np.ndarray,
    mode: str,
    groups: list[list[int]] | None = None,
) -> None:
    segments = np.load(dest_dir / "superpixels.npy", mmap_mode="r")
    num_segments = int(segments.max()) + 1
    x0, y0, x1, y1 = bbox_rows(segments, [{"segment_id": str(i)} for i in range(num_segments)])
    area = np.zeros(num_segments, dtype=np.float64)
    sum_x = np.zeros(num_segments, dtype=np.float64)
    sum_y = np.zeros(num_segments, dtype=np.float64)
    x_coords = np.arange(segments.shape[1], dtype=np.float64)
    for chunk_y0 in range(0, segments.shape[0], 512):
        chunk_y1 = min(segments.shape[0], chunk_y0 + 512)
        block = np.asarray(segments[chunk_y0:chunk_y1])
        valid = block >= 0
        if not np.any(valid):
            continue
        ids = block[valid].astype(np.int64, copy=False)
        yy = np.broadcast_to(np.arange(chunk_y0, chunk_y1, dtype=np.float64)[:, None], block.shape)[valid]
        xx = np.broadcast_to(x_coords[None, :], block.shape)[valid]
        area += np.bincount(ids, minlength=num_segments)
        sum_x += np.bincount(ids, weights=xx, minlength=num_segments)
        sum_y += np.bincount(ids, weights=yy, minlength=num_segments)
    cx = sum_x / np.maximum(area, 1)
    cy = sum_y / np.maximum(area, 1)

    rows: list[dict[str, object]] = []
    if mode == "split":
        parent_area = np.asarray([float(row["area"]) for row in medium_rows], dtype=np.float64)
        for segment_id, parent_id in enumerate(parent.tolist()):
            prow = medium_rows[parent_id]
            fraction = float(area[segment_id] / max(parent_area[parent_id], 1.0))
            count = int(round(float(prow["cell_count"]) * fraction))
            rows.append(make_standard_row(segment_id, area, cx, cy, x0, y0, x1, y1, count, prow))
    else:
        assert groups is not None
        for segment_id, members in enumerate(groups):
            label_area: Counter[int] = Counter()
            weighted_purity = 0.0
            weighted_valid = 0.0
            cell_count = 0
            for parent_id in members:
                prow = medium_rows[parent_id]
                parent_area = float(prow["area"])
                label_area[int(prow["gt_majority_label"])] += int(parent_area)
                weighted_purity += float(prow["gt_purity"]) * parent_area
                weighted_valid += float(prow["valid_fraction"]) * parent_area
                cell_count += int(prow["cell_count"])
            majority = label_area.most_common(1)[0][0] if label_area else 255
            pseudo = {
                "cell_count": cell_count,
                "gt_majority_label": majority,
                "gt_purity": weighted_purity / max(float(area[segment_id]), 1.0),
                "valid_fraction": weighted_valid / max(float(area[segment_id]), 1.0),
            }
            rows.append(make_standard_row(segment_id, area, cx, cy, x0, y0, x1, y1, cell_count, pseudo))
    write_csv(dest_dir / "superpixels.csv", rows, STANDARD_FIELDS)
    write_csv(
        dest_dir / "gt_label_stats.csv",
        gt_label_stats(rows),
        ["gt_majority_label", "segment_count", "valid_segment_count", "area", "mean_gt_purity", "mean_valid_fraction"],
    )


def make_standard_row(
    segment_id: int,
    area: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    cell_count: int,
    source: dict[str, object],
) -> dict[str, object]:
    segment_area = int(area[segment_id])
    return {
        "segment_id": segment_id,
        "area": segment_area,
        "center_x": float(cx[segment_id]),
        "center_y": float(cy[segment_id]),
        "bbox_x0": int(x0[segment_id]),
        "bbox_y0": int(y0[segment_id]),
        "bbox_x1": int(x1[segment_id]),
        "bbox_y1": int(y1[segment_id]),
        "cell_count": int(cell_count),
        "cell_density": float(cell_count / max(segment_area, 1)),
        "gt_majority_label": int(source["gt_majority_label"]),
        "gt_target_fraction": float(source["gt_purity"]),
        "gt_purity": float(source["gt_purity"]),
        "valid_fraction": float(source["valid_fraction"]),
    }


def link_or_copy_file(source: Path, dest: Path) -> None:
    ensure_link(dest, source)


def standardize_scale_output(
    image_id: str,
    scale: str,
    source_dir: Path,
    dest_dir: Path,
    factor: float,
    force: bool,
) -> dict[str, object]:
    validation_path = dest_dir / "validation.json"
    if not force and validation_path.exists():
        existing = json.loads(validation_path.read_text(encoding="utf-8"))
        if existing.get("passed"):
            return existing
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = read_csv(source_dir / "superpixels.csv")
    segments = np.load(source_dir / "superpixels.npy", mmap_mode="r")
    standard_rows = standardize_superpixels(records, segments)
    edges = adjacency_edges(segments)

    file_map = {
        "he_10x_rgb.npy": source_dir / "he_10x_rgb.npy",
        "he_tissue_mask.npy": source_dir / "he_tissue_mask.npy",
        "he_tissue_high_conf.npy": source_dir / "he_tissue_high_conf.npy",
        "he_tissue_low_conf.npy": source_dir / "he_tissue_low_conf.npy",
        "superpixels.npy": source_dir / "superpixels.npy",
        "tokens_image_cell_reg.npy": source_dir / "tokens_image_cell_reg_cellw0p5.npy",
        "tokens_image_cell_reg_cellw0p5.npy": source_dir / "tokens_image_cell_reg_cellw0p5.npy",
        "tokens_image_cell_reg_texture_cellstats.npy": source_dir / "tokens_image_cell_reg_texture_cellstats.npy",
    }
    for name, source in file_map.items():
        if not source.exists():
            continue
        if force or not (dest_dir / name).exists():
            link_or_copy_file(source, dest_dir / name)

    write_csv(dest_dir / "superpixels.csv", standard_rows, STANDARD_FIELDS)
    write_csv(
        dest_dir / "gt_label_stats.csv",
        gt_label_stats(standard_rows),
        ["gt_majority_label", "segment_count", "valid_segment_count", "area", "mean_gt_purity", "mean_valid_fraction"],
    )
    np.save(dest_dir / "adjacency.npy", edges)

    base = np.load(dest_dir / "tokens_image_cell_reg.npy", mmap_mode="r")
    enhanced = np.load(dest_dir / "tokens_image_cell_reg_texture_cellstats.npy", mmap_mode="r")
    validation_source = json.loads((source_dir / "validation.json").read_text(encoding="utf-8"))
    validation = {
        "passed": bool(
            validation_source.get("passed", False)
            and len(standard_rows) == int(base.shape[0]) == int(enhanced.shape[0])
            and edges.ndim == 2
            and edges.shape[1] == 2
        ),
        "image_id": image_id,
        "scale": scale,
        "scale_factor": factor,
        "source_dir": str(source_dir),
        "segment_count": len(standard_rows),
        "adjacency_edges": int(edges.shape[0]),
        "base_token_shape": list(base.shape),
        "enhanced_token_shape": list(enhanced.shape),
        "source_validation": validation_source,
    }
    write_json(dest_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"stage A validation failed for {image_id}/{scale}: {validation}")
    return validation


def derive_hierarchical_scale_output(
    image_id: str,
    scale: str,
    medium_dir: Path,
    dest_dir: Path,
    factor: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    validation_path = dest_dir / "validation.json"
    if not args.force and validation_path.exists():
        existing = json.loads(validation_path.read_text(encoding="utf-8"))
        if existing.get("passed"):
            return existing
    dest_dir.mkdir(parents=True, exist_ok=True)
    ensure_link(dest_dir / "he_10x_rgb.npy", medium_dir / "he_10x_rgb.npy")
    if scale == "small":
        write_small_from_medium(medium_dir, dest_dir, args.chunk_rows)
    elif scale == "large":
        write_large_from_medium(medium_dir, dest_dir, factor, args.chunk_rows)
    else:
        raise ValueError(scale)
    segments = np.load(dest_dir / "superpixels.npy", mmap_mode="r")
    edges = adjacency_edges(segments)
    np.save(dest_dir / "adjacency.npy", edges)
    base = np.load(dest_dir / "tokens_image_cell_reg.npy", mmap_mode="r")
    enhanced = np.load(dest_dir / "tokens_image_cell_reg_texture_cellstats.npy", mmap_mode="r")
    validation = {
        "passed": bool(base.shape[0] == enhanced.shape[0] == int(segments.max()) + 1 and edges.ndim == 2 and edges.shape[1] == 2),
        "image_id": image_id,
        "scale": scale,
        "scale_factor": factor,
        "source_dir": str(medium_dir),
        "derivation": "hierarchical_from_medium",
        "segment_count": int(segments.max()) + 1,
        "adjacency_edges": int(edges.shape[0]),
        "base_token_shape": list(base.shape),
        "enhanced_token_shape": list(enhanced.shape),
    }
    write_json(dest_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"stage A derived validation failed for {image_id}/{scale}: {validation}")
    return validation


def standardize_all(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    stage_root = args.output_root / "pret_superpixel"
    generated_root = stage_root / "_stage_a_generation"
    for row in rows:
        image_id = row["image_id"]
        source_by_scale = {
            "small": generated_root / "small" / "pret_superpixel" / image_id,
            "medium": generated_root / "medium" / "pret_superpixel" / image_id
            if args.generation_mode == "slic"
            else Path(row["existing_superpixel_dir"]),
            "large": generated_root / "large" / "pret_superpixel" / image_id,
        }
        for scale, factor in SCALES.items():
            dest = stage_root / "multiscale_tokens" / image_id / scale
            if (
                (args.generation_mode == "hierarchical" and scale in {"small", "large"})
                or (args.generation_mode == "large_slic" and scale == "small")
            ):
                output.append(derive_hierarchical_scale_output(image_id, scale, Path(row["existing_superpixel_dir"]), dest, factor, args))
            else:
                output.append(standardize_scale_output(image_id, scale, source_by_scale[scale], dest, factor, args.force))
            print(f"standardized {image_id}/{scale}", flush=True)
    return output


def write_stage_report(path: Path, validations: list[dict[str, object]], args: argparse.Namespace) -> None:
    by_scale = defaultdict(list)
    for row in validations:
        by_scale[str(row["scale"])].append(row)
    lines = [
        "# Phase A 多尺度 superpixel / region token 报告",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- output_root: `{args.output_root}`",
        f"- data_manifest: `{args.data_manifest}`",
        f"- generation_mode: {args.generation_mode}",
        f"- total_image_scale_outputs: {len(validations)}",
        f"- passed: {str(all(row['passed'] for row in validations)).lower()}",
        "",
        "## 尺度统计",
        "",
    ]
    for scale in ("small", "medium", "large"):
        group = by_scale[scale]
        segments = [int(row["segment_count"]) for row in group]
        edges = [int(row["adjacency_edges"]) for row in group]
        lines.extend(
            [
                f"### {scale}",
                "",
                f"- image_count: {len(group)}",
                f"- segment_count_min/median/max: {min(segments)}/{float(np.median(segments)):.1f}/{max(segments)}",
                f"- adjacency_edges_min/median/max: {min(edges)}/{float(np.median(edges)):.1f}/{max(edges)}",
                f"- base_token_shape_example: {group[0]['base_token_shape'] if group else []}",
                f"- enhanced_token_shape_example: {group[0]['enhanced_token_shape'] if group else []}",
                "",
            ]
        )
    lines.extend(
        [
            "## 输出说明",
            "",
            "- `multiscale_tokens/<image_id>/<scale>/tokens_image_cell_reg.npy` 是主方法 token，指向 `image_cell_reg_cellw0p5`。",
            "- `adjacency.npy` 保存无向邻接边列表，shape 为 `[edge_count, 2]`，不是稠密邻接矩阵。",
            "- `medium` 复用当前 full10x auto-physical 主尺度结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CellAtlas v3 phase A multiscale superpixel tokens.")
    parser.add_argument("--data_manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=V3_ROOT)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generation_mode", choices=["hierarchical", "large_slic", "slic"], default="hierarchical")
    parser.add_argument("--chunk_rows", type=int, default=512)
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
    parser.add_argument("--max_target_dimension", type=int, default=0)
    parser.add_argument("--max_texture_dimension", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.data_manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    if not rows:
        raise RuntimeError(f"empty data manifest: {args.data_manifest}")

    stage_root = args.output_root / "pret_superpixel"
    generated_root = stage_root / "_stage_a_generation"
    if args.generation_mode in {"large_slic", "slic"}:
        jobs = []
        if args.generation_mode == "slic":
            jobs.append(ScaleJob("small", 0.5, generated_root / "small", max(1, int(round(args.base_sp_diameter * 0.5)))))
            jobs.append(ScaleJob("medium", 1.0, generated_root / "medium", args.base_sp_diameter))
        jobs.append(ScaleJob("large", 2.0, generated_root / "large", max(1, int(round(args.base_sp_diameter * 2.0)))))
        for job in jobs:
            generate_scale(job, rows, args)

    validations = standardize_all(rows, args)
    summary = {
        "passed": all(row["passed"] for row in validations),
        "images": len(rows),
        "scales": sorted(SCALES),
        "image_scale_outputs": len(validations),
        "scale_counts": dict(Counter(str(row["scale"]) for row in validations)),
        "generation_mode": args.generation_mode,
        "validations": validations,
    }
    write_json(stage_root / "reports" / "phase_a_validation.json", summary)
    write_stage_report(REPORT_DIR / "report.md", validations, args)
    print(json.dumps({key: summary[key] for key in ("passed", "images", "scales", "image_scale_outputs", "scale_counts")}, indent=2))
    if not summary["passed"]:
        raise RuntimeError("phase A validation failed")


if __name__ == "__main__":
    main()
