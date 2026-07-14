# 作用：修复阶段 A 中 GT 非背景但没有 superpixel 覆盖的低饱和组织区域。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_a.build_multiscale_tokens import (  # noqa: E402
    STANDARD_FIELDS,
    adjacency_edges,
    gt_label_stats,
    write_csv,
    write_json,
)


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
DATA_MANIFEST = V3_ROOT / "pret_superpixel" / "data_manifest_v3.csv"
LEGACY_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2")
REPORT_DIR = Path(__file__).resolve().parent
SCALES = ["small", "medium", "large"]
BACKGROUND_LABEL = 2
NUM_CLASSES = 12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def load_gt(image_id: str) -> np.ndarray:
    return np.load(LEGACY_ROOT / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r")


def overwrite_npy(path: Path, values: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        np.save(file, values)
    os.replace(tmp, path)


def grid_fill_uncovered(seg: np.ndarray, gt: np.ndarray, grid_size: int) -> tuple[np.ndarray, int, int]:
    repaired = np.asarray(seg).copy()
    missing = (repaired < 0) & (gt != BACKGROUND_LABEL) & (gt >= 0) & (gt < NUM_CLASSES)
    missing_pixels = int(missing.sum())
    next_id = int(repaired.max()) + 1 if np.any(repaired >= 0) else 0
    added = 0
    if missing_pixels == 0:
        return repaired, added, missing_pixels

    flat = np.flatnonzero(missing)
    ys = flat // missing.shape[1]
    xs = flat % missing.shape[1]
    y_start = int(ys.min() // grid_size * grid_size)
    y_stop = int(min(missing.shape[0], (ys.max() // grid_size + 1) * grid_size))
    x_start = int(xs.min() // grid_size * grid_size)
    x_stop = int(min(missing.shape[1], (xs.max() // grid_size + 1) * grid_size))
    for y0 in range(y_start, y_stop, grid_size):
        y1 = min(missing.shape[0], y0 + grid_size)
        for x0 in range(x_start, x_stop, grid_size):
            x1 = min(missing.shape[1], x0 + grid_size)
            block = missing[y0:y1, x0:x1]
            if not np.any(block):
                continue
            view = repaired[y0:y1, x0:x1]
            view[block] = next_id
            next_id += 1
            added += 1
    return repaired, added, missing_pixels


def segment_stats(seg: np.ndarray, gt: np.ndarray, existing_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    num_segments = int(seg.max()) + 1 if np.any(seg >= 0) else 0
    area = np.zeros(num_segments, dtype=np.float64)
    sum_x = np.zeros(num_segments, dtype=np.float64)
    sum_y = np.zeros(num_segments, dtype=np.float64)
    x0 = np.full(num_segments, seg.shape[1], dtype=np.int64)
    y0 = np.full(num_segments, seg.shape[0], dtype=np.int64)
    x1 = np.zeros(num_segments, dtype=np.int64)
    y1 = np.zeros(num_segments, dtype=np.int64)
    counts = np.zeros((num_segments, NUM_CLASSES), dtype=np.int64)
    x_coords_float = np.arange(seg.shape[1], dtype=np.float64)
    x_coords_int = np.arange(seg.shape[1], dtype=np.int64)
    for y_start in range(0, seg.shape[0], 512):
        y_stop = min(seg.shape[0], y_start + 512)
        seg_block = seg[y_start:y_stop]
        valid = seg_block >= 0
        if not np.any(valid):
            continue
        ids = seg_block[valid].astype(np.int64, copy=False)
        yy_float = np.broadcast_to(np.arange(y_start, y_stop, dtype=np.float64)[:, None], seg_block.shape)[valid]
        xx_float = np.broadcast_to(x_coords_float[None, :], seg_block.shape)[valid]
        yy_int = yy_float.astype(np.int64, copy=False)
        xx_int = np.broadcast_to(x_coords_int[None, :], seg_block.shape)[valid]
        area += np.bincount(ids, minlength=num_segments)
        sum_x += np.bincount(ids, weights=xx_float, minlength=num_segments)
        sum_y += np.bincount(ids, weights=yy_float, minlength=num_segments)
        np.minimum.at(x0, ids, xx_int)
        np.minimum.at(y0, ids, yy_int)
        np.maximum.at(x1, ids, xx_int + 1)
        np.maximum.at(y1, ids, yy_int + 1)

        gt_block = gt[y_start:y_stop]
        valid_gt = valid & (gt_block >= 0) & (gt_block < NUM_CLASSES)
        if np.any(valid_gt):
            combined = (seg_block[valid_gt].astype(np.int64, copy=False) * NUM_CLASSES + gt_block[valid_gt].astype(np.int64, copy=False)).reshape(-1)
            counts += np.bincount(combined, minlength=num_segments * NUM_CLASSES).reshape(num_segments, NUM_CLASSES)
    labels = np.argmax(counts, axis=1).astype(np.int64)
    label_counts = counts[np.arange(num_segments), labels].astype(np.float64)
    valid_counts = counts.sum(axis=1).astype(np.float64)

    old_cell_counts = {
        int(row["segment_id"]): int(float(row.get("cell_count", 0)))
        for row in existing_rows
    }
    rows: list[dict[str, object]] = []
    for segment_id in range(num_segments):
        segment_area = int(area[segment_id])
        cell_count = old_cell_counts.get(segment_id, 0)
        purity = float(label_counts[segment_id] / max(valid_counts[segment_id], 1.0))
        valid_fraction = float(valid_counts[segment_id] / max(area[segment_id], 1.0))
        rows.append(
            {
                "segment_id": segment_id,
                "area": segment_area,
                "center_x": float(sum_x[segment_id] / max(area[segment_id], 1.0)),
                "center_y": float(sum_y[segment_id] / max(area[segment_id], 1.0)),
                "bbox_x0": int(x0[segment_id]) if segment_area else -1,
                "bbox_y0": int(y0[segment_id]) if segment_area else -1,
                "bbox_x1": int(x1[segment_id]) if segment_area else -1,
                "bbox_y1": int(y1[segment_id]) if segment_area else -1,
                "cell_count": cell_count,
                "cell_density": float(cell_count / max(segment_area, 1)),
                "gt_majority_label": int(labels[segment_id]) if valid_counts[segment_id] else 255,
                "gt_target_fraction": purity,
                "gt_purity": purity,
                "valid_fraction": valid_fraction,
            }
        )
    return rows


def append_tokens(scale_dir: Path, old_count: int, rows: list[dict[str, object]]) -> dict[str, list[int]]:
    if len(rows) == old_count:
        return {}
    from scipy.spatial import cKDTree

    old_centers = np.asarray([[float(row["center_x"]), float(row["center_y"])] for row in rows[:old_count]], dtype=np.float64)
    new_centers = np.asarray([[float(row["center_x"]), float(row["center_y"])] for row in rows[old_count:]], dtype=np.float64)
    _, nearest = cKDTree(old_centers).query(new_centers, k=1, workers=-1)
    nearest = nearest.astype(np.int64, copy=False)
    shapes: dict[str, list[int]] = {}
    for name in ["tokens_image_cell_reg.npy", "tokens_image_cell_reg_cellw0p5.npy", "tokens_image_cell_reg_texture_cellstats.npy"]:
        path = scale_dir / name
        token = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if token.shape[0] != old_count:
            raise RuntimeError(f"token row mismatch before repair: {path} token_rows={token.shape[0]} old_count={old_count}")
        repaired = np.concatenate([token, token[nearest]], axis=0)
        repaired = safe_l2_normalize(repaired).astype(np.float32)
        overwrite_npy(path, repaired)
        shapes[name] = list(repaired.shape)
    return shapes


def repair_image_scale(image_id: str, scale: str, args: argparse.Namespace) -> dict[str, object]:
    scale_dir = args.output_root / "pret_superpixel" / "multiscale_tokens" / image_id / scale
    seg_path = scale_dir / "superpixels.npy"
    old_rows = read_csv(scale_dir / "superpixels.csv")
    old_count = len(old_rows)
    seg = np.asarray(np.load(seg_path, mmap_mode="r"))
    gt = np.asarray(load_gt(image_id))
    if gt.shape != seg.shape:
        raise RuntimeError(f"GT/segment shape mismatch for {image_id}/{scale}: gt={gt.shape} seg={seg.shape}")

    areas = np.asarray([float(row["area"]) for row in old_rows if float(row["area"]) > 0], dtype=np.float64)
    grid_size = int(np.clip(round(np.sqrt(float(np.median(areas)))), args.min_grid_size, args.max_grid_size))
    repaired, added_segments, missing_pixels_before = grid_fill_uncovered(seg, gt, grid_size)
    missing_after = int(((repaired < 0) & (gt != BACKGROUND_LABEL) & (gt >= 0) & (gt < NUM_CLASSES)).sum())
    if added_segments:
        overwrite_npy(seg_path, repaired.astype(np.int32, copy=False))
    rows = segment_stats(repaired, gt, old_rows)
    token_shapes = append_tokens(scale_dir, old_count, rows)
    edges = adjacency_edges(repaired)
    np.save(scale_dir / "adjacency.npy", edges)
    write_csv(scale_dir / "superpixels.csv", rows, STANDARD_FIELDS)
    write_csv(
        scale_dir / "gt_label_stats.csv",
        gt_label_stats(rows),
        ["gt_majority_label", "segment_count", "valid_segment_count", "area", "mean_gt_purity", "mean_valid_fraction"],
    )

    base = np.load(scale_dir / "tokens_image_cell_reg.npy", mmap_mode="r")
    enhanced = np.load(scale_dir / "tokens_image_cell_reg_texture_cellstats.npy", mmap_mode="r")
    validation = json.loads((scale_dir / "validation.json").read_text(encoding="utf-8"))
    validation.update(
        {
            "passed": bool(missing_after == 0 and base.shape[0] == enhanced.shape[0] == len(rows)),
            "coverage_repair": "grid_fill_non_background_gt",
            "coverage_repair_background_label": BACKGROUND_LABEL,
            "coverage_repair_grid_size": grid_size,
            "coverage_missing_pixels_before": missing_pixels_before,
            "coverage_missing_pixels_after": missing_after,
            "coverage_added_segments": added_segments,
            "segment_count_before_repair": old_count,
            "segment_count": len(rows),
            "adjacency_edges": int(edges.shape[0]),
            "base_token_shape": list(base.shape),
            "enhanced_token_shape": list(enhanced.shape),
            "coverage_token_repair": "new segments inherit nearest existing segment token by center distance",
        }
    )
    write_json(scale_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"coverage repair failed for {image_id}/{scale}: {validation}")
    return {
        "image_id": image_id,
        "scale": scale,
        "grid_size": grid_size,
        "missing_pixels_before": missing_pixels_before,
        "missing_pixels_after": missing_after,
        "added_segments": added_segments,
        "segment_count_before": old_count,
        "segment_count_after": len(rows),
        "token_shapes": token_shapes,
    }


def write_report(path: Path, rows: list[dict[str, object]], output_root: Path) -> None:
    added = sum(int(row["added_segments"]) for row in rows)
    missing_before = sum(int(row["missing_pixels_before"]) for row in rows)
    missing_after = sum(int(row["missing_pixels_after"]) for row in rows)
    lines = [
        "# Phase A 覆盖修复报告",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- output_root: `{output_root}`",
        f"- repaired_image_scales: {len(rows)}",
        f"- added_segments: {added}",
        f"- missing_pixels_before: {missing_before}",
        f"- missing_pixels_after: {missing_after}",
        "",
        "## 修复方法",
        "",
        "- 只修复 `GT != background` 且 `superpixels == -1` 的区域。",
        "- 未覆盖区域按当前尺度典型 superpixel 尺寸切成 grid segment。",
        "- 新增 segment 的 token 继承最近已有 segment 的 token。",
        "- 背景区域保持 `-1`，不强行覆盖玻片空白。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair uncovered non-background GT tissue in phase A multiscale superpixels.")
    parser.add_argument("--data_manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=V3_ROOT)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--scales", nargs="+", default=SCALES, choices=SCALES)
    parser.add_argument("--min_grid_size", type=int, default=64)
    parser.add_argument("--max_grid_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.data_manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    if not rows:
        raise RuntimeError("no images selected")

    results: list[dict[str, object]] = []
    for row in rows:
        for scale in args.scales:
            result = repair_image_scale(row["image_id"], scale, args)
            results.append(result)
            print(
                f"repaired {row['image_id']}/{scale} "
                f"missing_before={result['missing_pixels_before']} added={result['added_segments']}",
                flush=True,
            )

    report = {
        "passed": all(int(row["missing_pixels_after"]) == 0 for row in results),
        "image_scales": len(results),
        "total_added_segments": sum(int(row["added_segments"]) for row in results),
        "total_missing_pixels_before": sum(int(row["missing_pixels_before"]) for row in results),
        "total_missing_pixels_after": sum(int(row["missing_pixels_after"]) for row in results),
        "results": results,
    }
    reports_dir = args.output_root / "pret_superpixel" / "reports"
    write_json(reports_dir / "phase_a_coverage_repair_validation.json", report)
    write_report(REPORT_DIR / "coverage_repair_report.md", results, args.output_root)
    print(json.dumps({key: report[key] for key in ["passed", "image_scales", "total_added_segments", "total_missing_pixels_before", "total_missing_pixels_after"]}, indent=2))
    if not report["passed"]:
        raise RuntimeError("coverage repair validation failed")


if __name__ == "__main__":
    main()
