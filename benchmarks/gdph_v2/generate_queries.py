from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def box_bounds(cx: float, cy: float, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = size / 2
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(width, x0 + size)
    y1 = min(height, y0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    return x0, y0, x1, y1


def generate_slide_queries(
    image_id: str,
    output_root: Path,
    box_sizes: list[int],
    per_class_size: int,
    min_purity: float,
    min_gt_valid_fraction: float,
    min_cells: int,
    acellular_classes: tuple[int, ...] = (3, 9, 10),
    acellular_sample_stride_gt: int = 128,
    acellular_candidate_pool: int = 64,
) -> list[dict]:
    slide_dir = output_root / "cells" / image_id
    cells = _read_csv(slide_dir / "cells.csv")
    labels = _read_csv(slide_dir / "tissue_labels.csv")
    if len(cells) != len(labels):
        raise RuntimeError(f"{image_id} cells/labels mismatch")
    expected_indices = list(range(len(cells)))
    if [int(row["cell_index"]) for row in cells] != expected_indices or [
        int(row["cell_index"]) for row in labels
    ] != expected_indices:
        raise RuntimeError(f"{image_id} cells/labels indices are not aligned")
    gt_mask = np.load(output_root / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r")
    gt_height, gt_width = gt_mask.shape
    with open(slide_dir / "validation.json", "r", encoding="utf-8") as file:
        slide_validation = json.load(file)
    original_width, original_height = slide_validation["original_size"]

    x = np.asarray([float(row["x_original"]) for row in cells])
    y = np.asarray([float(row["y_original"]) for row in cells])
    tissue = np.asarray([int(row["gt_tissue_label"]) for row in labels])
    valid = np.asarray([row["label_status"] == "valid" for row in labels])
    results = []
    for class_id in sorted(set(tissue[valid].tolist())):
        candidate_indices = np.flatnonzero(valid & (tissue == class_id))
        if candidate_indices.size == 0:
            continue
        candidate_indices = candidate_indices[
            np.argsort(np.asarray([cells[index]["cell_index"] for index in candidate_indices], dtype=int))
        ]
        for size in box_sizes:
            accepted_centers: list[tuple[float, float]] = []
            for index in candidate_indices:
                x0, y0, x1, y1 = box_bounds(x[index], y[index], size, original_width, original_height)
                inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & valid
                if int(inside.sum()) < min_cells:
                    continue
                gx0 = int(np.floor(x0 * gt_width / original_width))
                gy0 = int(np.floor(y0 * gt_height / original_height))
                gx1 = max(gx0 + 1, int(np.ceil(x1 * gt_width / original_width)))
                gy1 = max(gy0 + 1, int(np.ceil(y1 * gt_height / original_height)))
                region = np.asarray(gt_mask[gy0:gy1, gx0:gx1])
                valid_pixels = region[(region >= 0) & (region < 12)]
                if valid_pixels.size == 0:
                    continue
                gt_valid_fraction = float(valid_pixels.size / region.size)
                if gt_valid_fraction < min_gt_valid_fraction:
                    continue
                purity = float(np.mean(valid_pixels == class_id))
                if purity < min_purity:
                    continue
                if any((x[index] - px) ** 2 + (y[index] - py) ** 2 < size**2 for px, py in accepted_centers):
                    continue
                accepted_centers.append((x[index], y[index]))
                results.append(
                    {
                        "query_id": f"{image_id}_c{class_id}_s{size}_q{len(accepted_centers) - 1}",
                        "image_id": image_id,
                        "class_id": class_id,
                        "box_size_original": size,
                        "x0_original": x0,
                        "y0_original": y0,
                        "x1_original": x1,
                        "y1_original": y1,
                        "query_cells": int(inside.sum()),
                        "gt_purity": purity,
                        "gt_valid_fraction": gt_valid_fraction,
                        "query_mode": "cell_seeded",
                        "cell_query_eligible": True,
                    }
                )
                if len(accepted_centers) >= per_class_size:
                    break

    # Explicitly sample low-cell tissues without requiring a nucleus seed.
    # These queries expose where cell-only retrieval is undefined while patch
    # or hybrid representations can still be evaluated.
    sampled_gt = np.asarray(gt_mask[::acellular_sample_stride_gt, ::acellular_sample_stride_gt])
    for class_id in acellular_classes:
        sampled_yx = np.argwhere(sampled_gt == class_id)
        if len(sampled_yx) == 0:
            continue
        for size in box_sizes:
            seed_bytes = hashlib.sha256(f"{image_id}|{class_id}|{size}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "little"))
            candidate_order = rng.permutation(len(sampled_yx))
            candidates = []
            for candidate_index in candidate_order:
                sample_y, sample_x = sampled_yx[candidate_index]
                center_gt_x = (float(sample_x) + 0.5) * acellular_sample_stride_gt
                center_gt_y = (float(sample_y) + 0.5) * acellular_sample_stride_gt
                center_x = center_gt_x * original_width / gt_width
                center_y = center_gt_y * original_height / gt_height
                x0, y0, x1, y1 = box_bounds(
                    center_x, center_y, size, original_width, original_height
                )
                gx0 = int(np.floor(x0 * gt_width / original_width))
                gy0 = int(np.floor(y0 * gt_height / original_height))
                gx1 = max(gx0 + 1, int(np.ceil(x1 * gt_width / original_width)))
                gy1 = max(gy0 + 1, int(np.ceil(y1 * gt_height / original_height)))
                region = np.asarray(gt_mask[gy0:gy1, gx0:gx1])
                valid_pixels = region[(region >= 0) & (region < 12)]
                if valid_pixels.size == 0:
                    continue
                gt_valid_fraction = float(valid_pixels.size / region.size)
                purity = float(np.mean(valid_pixels == class_id))
                if purity < min_purity or gt_valid_fraction < min_gt_valid_fraction:
                    continue
                inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & valid
                candidates.append(
                    {
                        "center": (center_x, center_y),
                        "bounds": (x0, y0, x1, y1),
                        "query_cells": int(inside.sum()),
                        "gt_purity": purity,
                        "gt_valid_fraction": gt_valid_fraction,
                    }
                )
                if len(candidates) >= acellular_candidate_pool:
                    break
            accepted = []
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item["query_cells"],
                    item["center"][1],
                    item["center"][0],
                ),
            ):
                cx, cy = candidate["center"]
                if any((cx - px) ** 2 + (cy - py) ** 2 < size**2 for px, py in accepted):
                    continue
                accepted.append((cx, cy))
                x0, y0, x1, y1 = candidate["bounds"]
                results.append(
                    {
                        "query_id": f"{image_id}_c{class_id}_s{size}_a{len(accepted) - 1}",
                        "image_id": image_id,
                        "class_id": class_id,
                        "box_size_original": size,
                        "x0_original": x0,
                        "y0_original": y0,
                        "x1_original": x1,
                        "y1_original": y1,
                        "query_cells": candidate["query_cells"],
                        "gt_purity": candidate["gt_purity"],
                        "gt_valid_fraction": candidate["gt_valid_fraction"],
                        "query_mode": "gt_screened_low_cell",
                        "cell_query_eligible": candidate["query_cells"] >= min_cells,
                    }
                )
                if len(accepted) >= per_class_size:
                    break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pure automatic GDPH region queries.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--box_sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--queries_per_class_size", type=int, default=3)
    parser.add_argument("--min_purity", type=float, default=0.8)
    parser.add_argument("--min_gt_valid_fraction", type=float, default=0.8)
    parser.add_argument("--min_cells", type=int, default=5)
    args = parser.parse_args()
    if not (0 <= args.min_purity <= 1 and 0 <= args.min_gt_valid_fraction <= 1):
        raise ValueError("purity and GT valid fraction thresholds must be in [0, 1]")
    rows = _read_csv(args.manifest)
    queries = []
    for row in rows:
        queries.extend(
            generate_slide_queries(
                row["image_id"],
                Path(args.output_root),
                args.box_sizes,
                args.queries_per_class_size,
                args.min_purity,
                args.min_gt_valid_fraction,
                args.min_cells,
            )
        )
    output_path = Path(args.output_root) / "region_retrieval" / "queries.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not queries:
        raise RuntimeError("query generation produced no regions")
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(queries[0]))
        writer.writeheader()
        writer.writerows(queries)
    temporary_output.replace(output_path)
    summary = defaultdict(int)
    for query in queries:
        summary[(query["class_id"], query["box_size_original"])] += 1
    validation = {
        "images_in_manifest": len(rows),
        "queries": len(queries),
        "unique_query_ids": len({query["query_id"] for query in queries}),
        "classes": sorted({int(query["class_id"]) for query in queries}),
        "box_sizes_original": sorted({int(query["box_size_original"]) for query in queries}),
        "min_purity": args.min_purity,
        "min_gt_valid_fraction": args.min_gt_valid_fraction,
        "min_cells": args.min_cells,
        "selection_protocol": (
            "deterministic GT-screened automatic simulation of a user-drawn box, "
            "including explicit low-cell necrosis/mucus/fat boxes; "
            "GT is used only to choose/evaluate queries, not to compute retrieval scores"
        ),
        "all_thresholds_satisfied": all(
            float(query["gt_purity"]) >= args.min_purity
            and float(query["gt_valid_fraction"]) >= args.min_gt_valid_fraction
            and (
                query["query_mode"] == "gt_screened_low_cell"
                or int(query["query_cells"]) >= args.min_cells
            )
            for query in queries
        ),
        "low_cell_queries": sum(
            query["query_mode"] == "gt_screened_low_cell" for query in queries
        ),
        "zero_cell_queries": sum(int(query["query_cells"]) == 0 for query in queries),
        "cell_ineligible_low_cell_queries": sum(
            query["query_mode"] == "gt_screened_low_cell"
            and not bool(query["cell_query_eligible"])
            for query in queries
        ),
        "low_cell_query_classes": sorted(
            {
                int(query["class_id"])
                for query in queries
                if query["query_mode"] == "gt_screened_low_cell"
            }
        ),
    }
    validation["passed"] = bool(
        validation["images_in_manifest"] == 20
        and validation["queries"] == validation["unique_query_ids"] > 0
        and validation["box_sizes_original"] == sorted(args.box_sizes)
        and validation["all_thresholds_satisfied"]
        and validation["low_cell_queries"] > 0
        and validation["cell_ineligible_low_cell_queries"] > 0
        and 10 in validation["low_cell_query_classes"]
    )
    validation_path = output_path.parent / "queries_validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_validation.replace(validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"query validation failed: {validation}")
    print(output_path, "queries=", len(queries))
    print(json.dumps({f"class_{key[0]}_size_{key[1]}": value for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
