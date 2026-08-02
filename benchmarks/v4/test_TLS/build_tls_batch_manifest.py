#!/usr/bin/env python3
"""Build a deterministic random manifest of valid annotated TLS WSIs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from benchmarks.v4.test_TLS.prepare_tls_case import load_polygons
from benchmarks.v4.whole_slide_inference.src.tiling import build_tile_rows
from module.KFBreader.kfbreader import KFBSlide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-annotations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/test_TLS"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def polygon_area(polygon: np.ndarray) -> float:
    return abs(float(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    )) / 2.0


def deterministic_case_key(case_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:case:{case_id}".encode("utf-8")).digest()


def deterministic_prompt_indices(case_id: str, count: int, seed: int) -> list[int]:
    prompt_count = count // 2
    ranked = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"{seed}:prompt:{case_id}:{index}".encode("utf-8")
        ).digest(),
    )
    return sorted(ranked[:prompt_count])


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.min_annotations < 1:
        raise ValueError("count and min-annotations must be positive")
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"tls_batch_manifest_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    slide_root = args.dataset_root / "950例肠癌HE图像"
    annotation_root = args.dataset_root / "标注文件"
    slides = sorted(slide_root.glob("*.kfb"))
    annotation_dirs = [path for path in annotation_root.iterdir() if path.is_dir()]
    by_case: dict[str, list[Path]] = {}
    for path in annotation_dirs:
        by_case.setdefault(path.name.split()[0].split("_")[0], []).append(path)

    eligible: list[dict] = []
    skipped: list[dict] = []
    for dataset_ordinal, wsi_path in enumerate(slides, start=1):
        case_id = wsi_path.name.split()[0]
        candidates = [
            path / "Annotations" / "1.json" for path in by_case.get(case_id, [])
            if (path / "Annotations" / "1.json").is_file()
        ]
        if len(candidates) != 1:
            skipped.append({
                "dataset_ordinal": dataset_ordinal, "case_id": case_id,
                "reason": f"annotation_json_candidates={len(candidates)}",
            })
            continue
        annotation_json = candidates[0]
        try:
            polygons = load_polygons(annotation_json)
            if len(polygons) < args.min_annotations:
                raise ValueError(
                    f"annotation_count={len(polygons)} < min_annotations={args.min_annotations}"
                )
        except Exception as exc:
            skipped.append({
                "dataset_ordinal": dataset_ordinal, "case_id": case_id,
                "reason": repr(exc),
            })
            continue
        eligible.append({
            "dataset_ordinal": dataset_ordinal,
            "case_id": case_id,
            "wsi_path": wsi_path,
            "annotation_json": annotation_json,
            "polygons": polygons,
        })

    eligible.sort(key=lambda item: deterministic_case_key(item["case_id"], args.seed))
    rows: list[dict] = []
    validation_failures: list[dict] = []
    for candidate in eligible:
        dataset_ordinal = candidate["dataset_ordinal"]
        case_id = candidate["case_id"]
        wsi_path = candidate["wsi_path"]
        annotation_json = candidate["annotation_json"]
        polygons = candidate["polygons"]
        try:
            slide = KFBSlide(str(wsi_path))
            width, height = map(int, slide.dimensions)
            if any(
                np.any(poly[:, 0] < 0) or np.any(poly[:, 0] >= width)
                or np.any(poly[:, 1] < 0) or np.any(poly[:, 1] >= height)
                for poly in polygons
            ):
                raise ValueError("annotation polygon lies outside WSI level-0 bounds")
            areas = np.asarray([polygon_area(poly) for poly in polygons], np.float64)
            if np.any(areas <= 0) or not np.isfinite(areas).all():
                raise ValueError("annotation contains non-positive polygon area")
            positive_indices = deterministic_prompt_indices(case_id, len(polygons), args.seed)
            wsi_id = f"HZ_TLS_{case_id}"
            tile_count = len(build_tile_rows(
                wsi_id, str(wsi_path), width, height, 4, 512, 384, "test"
            ))
        except Exception as exc:
            validation_failures.append({
                "dataset_ordinal": dataset_ordinal, "case_id": case_id,
                "reason": repr(exc),
            })
            continue
        rows.append({
            "selection_rank": len(rows) + 1,
            "dataset_ordinal": dataset_ordinal,
            "case_id": case_id,
            "wsi_id": wsi_id,
            "wsi_path": str(wsi_path),
            "annotation_json": str(annotation_json),
            "annotation_count": len(polygons),
            "positive_polygon_indices_json": json.dumps(positive_indices),
            "prompted_polygon_count": len(positive_indices),
            "held_out_polygon_count": len(polygons) - len(positive_indices),
            "prompted_polygon_area_level0": float(areas[positive_indices].sum()),
            "width_level0": width,
            "height_level0": height,
            "tile_count": tile_count,
        })
        if len(rows) == args.count:
            break
    if len(rows) != args.count:
        raise RuntimeError(f"found only {len(rows)} valid annotated WSIs, expected {args.count}")

    output.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    manifest = output / "tls_wsi_manifest.parquet"
    frame.to_parquet(manifest, index=False)
    summary = {
        "timestamp": stamp,
        "dataset_root": str(args.dataset_root),
        "selection_rule": (
            "uniform deterministic pseudo-random sample from all eligible WSIs, "
            "ordered by SHA256(seed:case:case_id)"
        ),
        "positive_prompt_rule": (
            "floor(annotation_count/2) TLS instances selected deterministically by "
            "SHA256(seed:prompt:case_id:polygon_index); one deepest-interior point per instance"
        ),
        "negative_prompt_rule": "three deterministic tissue points outside dilated TLS union",
        "prompt_seed": args.seed,
        "min_annotations": args.min_annotations,
        "split": "test",
        "eligible_wsi_count": len(eligible),
        "count": len(frame),
        "last_dataset_ordinal": int(frame.dataset_ordinal.max()),
        "annotation_total": int(frame.annotation_count.sum()),
        "annotation_min": int(frame.annotation_count.min()),
        "annotation_median": float(frame.annotation_count.median()),
        "annotation_mean": float(frame.annotation_count.mean()),
        "annotation_max": int(frame.annotation_count.max()),
        "prompted_annotation_total": int(frame.prompted_polygon_count.sum()),
        "held_out_annotation_total": int(frame.held_out_polygon_count.sum()),
        "held_out_annotation_min": int(frame.held_out_polygon_count.min()),
        "held_out_annotation_median": float(frame.held_out_polygon_count.median()),
        "held_out_annotation_max": int(frame.held_out_polygon_count.max()),
        "wsi_with_no_held_out_tls": int((frame.held_out_polygon_count == 0).sum()),
        "tile_total": int(frame.tile_count.sum()),
        "tile_min": int(frame.tile_count.min()),
        "tile_mean": float(frame.tile_count.mean()),
        "tile_max": int(frame.tile_count.max()),
        "manifest": str(manifest),
        "screening_failures": skipped,
        "validation_failures_before_selection_complete": validation_failures,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frame.to_csv(output / "tls_wsi_manifest.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
