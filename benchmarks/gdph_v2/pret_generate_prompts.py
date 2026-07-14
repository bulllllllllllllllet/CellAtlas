from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    box_original_to_target,
    format_segment_ids,
    parse_segment_ids,
    read_csv,
    write_csv_atomic,
    write_json_atomic,
)


def overlap_segments(
    segments: np.ndarray,
    box_10x: tuple[int, int, int, int],
    min_overlap_fraction: float,
    segment_area: np.ndarray | None = None,
) -> list[int]:
    x0, y0, x1, y1 = box_10x
    region = segments[y0:y1, x0:x1]
    ids, counts = np.unique(region[region >= 0], return_counts=True)
    if segment_area is None:
        segment_area = np.bincount(segments[segments >= 0].reshape(-1))
    output = []
    for segment_id, inside_count in zip(ids.tolist(), counts.tolist()):
        total = int(segment_area[int(segment_id)]) if int(segment_id) < len(segment_area) else 0
        if total and inside_count / total >= min_overlap_fraction:
            output.append(int(segment_id))
    return output


def overlap_segment_fractions(
    segments: np.ndarray,
    box_10x: tuple[int, int, int, int],
    segment_area: np.ndarray,
) -> dict[int, float]:
    x0, y0, x1, y1 = box_10x
    region = segments[y0:y1, x0:x1]
    ids, counts = np.unique(region[region >= 0], return_counts=True)
    output: dict[int, float] = {}
    for segment_id, inside_count in zip(ids.tolist(), counts.tolist()):
        total = int(segment_area[int(segment_id)]) if int(segment_id) < len(segment_area) else 0
        if total:
            output[int(segment_id)] = float(inside_count / total)
    return output


def centered_scribble_segments(
    records: list[dict[str, str]], box_10x: tuple[int, int, int, int], radius_fraction: float
) -> list[int]:
    x0, y0, x1, y1 = box_10x
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    radius = max(8.0, min(x1 - x0, y1 - y0) * radius_fraction)
    selected = []
    for row in records:
        dx = float(row["center_x_10x"]) - cx
        dy = float(row["center_y_10x"]) - cy
        if dx * dx + dy * dy <= radius * radius:
            selected.append(int(row["segment_id"]))
    return selected


def make_prompt_rows(
    output_root: Path,
    query: dict[str, str],
    shots: list[int],
    min_box_overlap: float,
    negative_multiplier: int,
    realistic_negative_source: str,
    negative_mode: str,
    negative_counts: list[int],
    prompt_sources: set[str],
    gt_pure_min_purity: float,
    image_cache: dict[str, dict[str, object]] | None = None,
) -> list[dict]:
    image_id = query["image_id"]
    if image_cache is not None and image_id in image_cache:
        cached = image_cache[image_id]
        records = cached["records"]
        original_size = cached["original_size"]
        segments = cached["segments"]
        segment_area = cached["segment_area"]
    else:
        superpixel_dir = output_root / PRET_DIR / image_id
        records = read_csv(superpixel_dir / "superpixels.csv")
        validation = json.loads((output_root / "cells" / image_id / "validation.json").read_text(encoding="utf-8"))
        original_size = tuple(int(value) for value in validation["original_size"])
        segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
        segment_area = np.bincount(np.asarray(segments)[np.asarray(segments) >= 0].reshape(-1))
        if image_cache is not None:
            image_cache[image_id] = {
                "records": records,
                "original_size": original_size,
                "segments": segments,
                "segment_area": segment_area,
            }
    box = tuple(float(query[key]) for key in ("x0_original", "y0_original", "x1_original", "y1_original"))
    box_10x = box_original_to_target(box, original_size, segments.shape)
    class_id = int(query["class_id"])
    overlap_fractions = overlap_segment_fractions(segments, box_10x, segment_area)
    inside = set(overlap_segments(segments, box_10x, min_box_overlap, segment_area))
    if not inside:
        return []

    by_id = {int(row["segment_id"]): row for row in records}
    x0, y0, x1, y1 = box_10x
    box_center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)

    def gt_pure_sort_key(segment_id: int) -> tuple[float, float, float, int, int]:
        row = by_id[segment_id]
        center = np.asarray([float(row["center_x_10x"]), float(row["center_y_10x"])], dtype=np.float64)
        return (
            -float(row["gt_label_purity"]),
            -float(overlap_fractions.get(segment_id, 0.0)),
            float(np.linalg.norm(center - box_center)),
            -int(row["area_10x_pixels"]),
            segment_id,
        )

    oracle_positive = [
        segment_id
        for segment_id in inside
        if int(by_id[segment_id]["gt_tissue_label"]) == class_id
        and by_id[segment_id]["valid_for_retrieval"].lower() == "true"
    ]
    gt_pure_positive = sorted(
        [
            segment_id
            for segment_id in inside
            if int(by_id[segment_id]["gt_tissue_label"]) == class_id
            and by_id[segment_id]["valid_for_retrieval"].lower() == "true"
            and float(by_id[segment_id]["gt_label_purity"]) >= gt_pure_min_purity
        ],
        key=gt_pure_sort_key,
    )
    realistic_positive = sorted(inside)
    scribble_positive = [
        segment_id for segment_id in centered_scribble_segments(records, box_10x, 0.2) if segment_id in inside
    ]
    candidates_by_source = {
        "oracle_gt_purity": oracle_positive,
        "gt_pure_box": gt_pure_positive,
        "realistic_box": realistic_positive,
        "scribble_like": scribble_positive or realistic_positive[:1],
    }
    negative_oracle = [
        int(row["segment_id"])
        for row in records
        if int(row["segment_id"]) not in inside
        and int(row["gt_tissue_label"]) != class_id
        and row["valid_for_retrieval"].lower() == "true"
    ]
    # A real box prompt does not tell the system which outside segments are
    # negatives. Treat outside-box negatives as a separate ablation because
    # same-class tissue can legitimately exist elsewhere in the slide.
    negative_realistic = (
        [
            int(row["segment_id"])
            for row in records
            if int(row["segment_id"]) not in inside and bool(int(row["area_10x_pixels"]) > 0)
        ]
        if realistic_negative_source == "outside_box"
        else []
    )
    rows = []

    def append_prompt(
        source: str,
        positive: list[int],
        negative_pool: list[int],
        requested_negative_counts: list[int] | None,
    ) -> None:
        if source == "gt_pure_box":
            positive = list(dict.fromkeys(positive))
        else:
            positive = sorted(set(positive))
        if not positive:
            return
        for shot in shots:
            chosen_positive = positive[: min(shot, len(positive))]
            if not chosen_positive:
                continue
            if requested_negative_counts is None:
                neg_counts = [max(shot, len(chosen_positive) * negative_multiplier)]
            else:
                neg_counts = requested_negative_counts
            for neg_count in neg_counts:
                chosen_negative = negative_pool[: min(neg_count, len(negative_pool))]
                purities = [float(by_id[segment_id]["gt_label_purity"]) for segment_id in chosen_positive if segment_id in by_id]
                areas = [float(by_id[segment_id]["area_10x_pixels"]) for segment_id in chosen_positive if segment_id in by_id]
                target_areas = [
                    float(by_id[segment_id]["area_10x_pixels"])
                    if int(by_id[segment_id]["gt_tissue_label"]) == class_id
                    else 0.0
                    for segment_id in chosen_positive
                    if segment_id in by_id
                ]
                valid_areas = [
                    float(by_id[segment_id]["area_10x_pixels"]) * float(by_id[segment_id]["gt_valid_fraction"])
                    for segment_id in chosen_positive
                    if segment_id in by_id
                ]
                total_area = float(np.sum(areas)) if areas else 0.0
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "image_id": image_id,
                        "class_id": class_id,
                        "shot": len(chosen_positive),
                        "prompt_source": source,
                        "prompt_mode": "positive_negative" if chosen_negative else "positive_only",
                        "x0_original": query["x0_original"],
                        "y0_original": query["y0_original"],
                        "x1_original": query["x1_original"],
                        "y1_original": query["y1_original"],
                        "positive_segment_count": len(chosen_positive),
                        "negative_segment_count": len(chosen_negative),
                        "prompt_purity": float(np.mean(purities)) if purities else 0.0,
                        "prompt_area_10x_pixels": total_area,
                        "prompt_target_area_fraction": float(np.sum(target_areas) / total_area) if total_area else 0.0,
                        "prompt_valid_area_fraction": float(np.sum(valid_areas) / total_area) if total_area else 0.0,
                        "positive_segments": format_segment_ids(chosen_positive),
                        "negative_segments": format_segment_ids(chosen_negative),
                        "gt_pure_min_purity": gt_pure_min_purity if source == "gt_pure_box" else "",
                        "box_candidate_segment_count": len(inside),
                        "gt_pure_candidate_segment_count": len(gt_pure_positive),
                    }
                )

    for source, positive in candidates_by_source.items():
        if source not in prompt_sources:
            continue
        negative_pool = negative_oracle if source == "oracle_gt_purity" else negative_realistic
        append_prompt(source, positive, negative_pool, None)
    if negative_mode == "oracle_contrast":
        append_prompt(
            "oracle_positive_negative",
            realistic_positive,
            negative_oracle,
            negative_counts,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRET-style simulated superpixel prompts.")
    parser.add_argument("--queries_csv", default=str(DEFAULT_OUTPUT_ROOT / "region_retrieval" / "queries.csv"))
    parser.add_argument(
        "--canonical_prompts_csv",
        default=None,
        help="Optional scale-stable prompt definition CSV. Final prompts are restricted to its query_id/source/shot rows.",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--prompts_output_csv",
        default=None,
        help="Optional output CSV for generated prompts. Defaults to <output_root>/pret_superpixel/prompts.csv.",
    )
    parser.add_argument(
        "--prompt_sources",
        nargs="+",
        choices=["oracle_gt_purity", "realistic_box", "scribble_like", "gt_pure_box"],
        default=["oracle_gt_purity", "realistic_box", "scribble_like"],
    )
    parser.add_argument("--gt_pure_min_purity", type=float, default=0.85)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--min_box_overlap", type=float, default=0.5)
    parser.add_argument("--negative_multiplier", type=int, default=3)
    parser.add_argument(
        "--realistic_negative_source",
        choices=["none", "outside_box"],
        default="none",
        help="Use 'outside_box' only as an ablation; it can sample same-class tissue as negatives.",
    )
    parser.add_argument("--negative_mode", choices=["none", "oracle_contrast"], default="none")
    parser.add_argument("--negative_counts", nargs="+", type=int, default=[1, 3])
    args = parser.parse_args()
    canonical_rows = read_csv(args.canonical_prompts_csv) if args.canonical_prompts_csv else []
    if canonical_rows:
        grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in canonical_rows:
            grouped[
                (
                    row["query_id"],
                    row["image_id"],
                    str(row["class_id"]),
                    row["x0_original"],
                    row["y0_original"],
                    row["x1_original"],
                    row["y1_original"],
                )
            ].append(row)
        queries = [
            {
                "query_id": key[0],
                "image_id": key[1],
                "class_id": key[2],
                "x0_original": key[3],
                "y0_original": key[4],
                "x1_original": key[5],
                "y1_original": key[6],
            }
            for key in grouped
        ]
        allowed = {
            (row["query_id"], row["prompt_source"], int(row["shot"]))
            for row in canonical_rows
        }
        requested_keys = set(allowed)
    else:
        queries = read_csv(args.queries_csv)
        allowed = set()
        requested_keys = set()
    if args.image_id:
        requested = set(args.image_id)
        queries = [query for query in queries if query["image_id"] in requested]
        if canonical_rows:
            requested_keys = {key for key in requested_keys if any(row["query_id"] == key[0] and row["image_id"] in requested for row in canonical_rows)}
    output_root = Path(args.output_root)
    prompt_sources = set(args.prompt_sources)
    image_cache: dict[str, dict[str, object]] = {}
    rows = []
    for query in queries:
        superpixel_dir = output_root / PRET_DIR / query["image_id"]
        if (superpixel_dir / "superpixels.csv").is_file():
            query_shots = args.shots
            if canonical_rows:
                query_shots = sorted({shot for query_id, _, shot in allowed if query_id == query["query_id"]})
            generated = make_prompt_rows(
                output_root,
                query,
                query_shots,
                args.min_box_overlap,
                args.negative_multiplier,
                args.realistic_negative_source,
                args.negative_mode,
                args.negative_counts,
                prompt_sources,
                args.gt_pure_min_purity,
                image_cache,
            )
            if canonical_rows:
                generated = [
                    row for row in generated
                    if (row["query_id"], row["prompt_source"], int(row["shot"])) in allowed
                ]
            rows.extend(generated)
    if not rows:
        raise RuntimeError("no PRET prompts generated")
    output_path = Path(args.prompts_output_csv) if args.prompts_output_csv else output_root / PRET_DIR / "prompts.csv"
    output_dir = output_path.parent
    write_csv_atomic(output_path, rows)
    produced_keys = {
        (row["query_id"], row["prompt_source"], int(row["shot"]))
        for row in rows
    }
    dropped = []
    if canonical_rows:
        by_key = {
            (row["query_id"], row["prompt_source"], int(row["shot"])): row
            for row in canonical_rows
        }
        for key in sorted(requested_keys - produced_keys):
            row = dict(by_key[key])
            row["drop_reason"] = "no_scale_mapped_positive_prompt"
            dropped.append(row)
        if dropped:
            write_csv_atomic(output_dir / "dropped_prompts.csv", dropped)
    validation = {
        "passed": True,
        "prompts": len(rows),
        "canonical_prompts_csv": args.canonical_prompts_csv or "",
        "requested_canonical_prompts": len(requested_keys) if canonical_rows else 0,
        "dropped_canonical_prompts": len(dropped),
        "prompt_sources": sorted({row["prompt_source"] for row in rows}),
        "requested_prompt_sources": sorted(prompt_sources),
        "shots": sorted({int(row["shot"]) for row in rows}),
        "oracle_prompts": sum(row["prompt_source"] == "oracle_gt_purity" for row in rows),
        "realistic_prompts": sum(row["prompt_source"] != "oracle_gt_purity" for row in rows),
        "gt_pure_prompts": sum(row["prompt_source"] == "gt_pure_box" for row in rows),
        "gt_pure_min_purity": args.gt_pure_min_purity,
        "prompts_csv": str(output_path),
        "realistic_negative_source": args.realistic_negative_source,
        "negative_mode": args.negative_mode,
        "negative_counts": args.negative_counts,
    }
    write_json_atomic(output_dir / "prompts_validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
