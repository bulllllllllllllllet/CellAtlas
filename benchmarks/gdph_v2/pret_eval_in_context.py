from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.eval_retrieval import ranking_metrics
from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    area_sweep_metrics,
    binary_segmentation_metrics,
    connected_component_topk_mask,
    mean_std_threshold,
    normalized_median,
    otsu_score_threshold,
    parse_segment_ids,
    percentile_threshold,
    precision_at_area,
    prompt_relative_threshold,
    predict_top_area,
    read_csv,
    safe_l2_normalize,
    segment_adjacency,
    smooth_scores,
    stable_seed,
    write_csv_atomic,
    write_json_atomic,
)


NEIGHBOR_CACHE: dict[tuple[str, str], list[np.ndarray]] = {}
DEPLOYABLE_MASKS = (
    "otsu",
    "mean_std_0p5",
    "mean_std_1p0",
    "percentile_85",
    "percentile_90",
    "percentile_95",
    "prompt_relative_margin_0p05",
    "prompt_relative_margin_0p10",
    "cc_top20_keep3",
)
SUMMARY_EXTRA_FIELDS = (
    "calibrated_top18_area_dice",
    "calibrated_percentile_90_dice",
    *(f"{name}_dice" for name in DEPLOYABLE_MASKS),
)


def scores_from_prompt(tokens: np.ndarray, positive: list[int], negative: list[int], strategy: str, lambda_neg: float) -> np.ndarray:
    normalized = safe_l2_normalize(tokens, axis=1)
    q_pos = normalized_median(normalized[positive])
    pos_score = normalized @ q_pos
    if not negative:
        return pos_score
    neg_tokens = safe_l2_normalize(normalized[negative], axis=1)
    if strategy == "negative_median":
        q_neg = normalized_median(neg_tokens)
        neg_score = normalized @ q_neg
    elif strategy == "negative_bank_max":
        neg_score = (normalized @ neg_tokens.T).max(axis=1)
    else:
        raise ValueError(f"unknown negative strategy: {strategy}")
    return pos_score - lambda_neg * neg_score


def candidate_neighbors_from_global(global_neighbors: list[np.ndarray], candidate: np.ndarray) -> list[np.ndarray]:
    global_indices = np.flatnonzero(candidate)
    local_index = {int(global_id): local_id for local_id, global_id in enumerate(global_indices)}
    output: list[np.ndarray] = []
    for global_id in global_indices:
        adjacent = [
            local_index[int(item)]
            for item in global_neighbors[int(global_id)]
            if int(item) in local_index
        ]
        output.append(np.asarray(adjacent, dtype=np.int64))
    return output


def threshold_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    positive_scores: np.ndarray,
    candidate_neighbors: list[np.ndarray],
) -> dict[str, float]:
    output: dict[str, float] = {}
    protocols: dict[str, np.ndarray] = {
        "otsu": scores >= otsu_score_threshold(scores),
        "mean_std_0p5": scores >= mean_std_threshold(scores, 0.5),
        "mean_std_1p0": scores >= mean_std_threshold(scores, 1.0),
        "percentile_85": scores >= percentile_threshold(scores, 85.0),
        "percentile_90": scores >= percentile_threshold(scores, 90.0),
        "percentile_95": scores >= percentile_threshold(scores, 95.0),
        "prompt_relative_margin_0p05": scores >= prompt_relative_threshold(scores, positive_scores, 0.05),
        "prompt_relative_margin_0p10": scores >= prompt_relative_threshold(scores, positive_scores, 0.10),
        "calibrated_top18_area": predict_top_area(scores, areas, 0.18),
        "calibrated_percentile_90": scores >= percentile_threshold(scores, 90.0),
    }
    if candidate_neighbors:
        protocols["cc_top20_keep3"] = connected_component_topk_mask(scores, areas, candidate_neighbors, 0.20, 3)
    else:
        protocols["cc_top20_keep3"] = predict_top_area(scores, areas, 0.20)
    for name, pred in protocols.items():
        metrics = binary_segmentation_metrics(target, pred)
        output[f"{name}_dice"] = float(metrics["dice"])
        output[f"{name}_iou"] = float(metrics["iou"])
        output[f"{name}_pred_area_fraction"] = float(np.sum(areas[pred]) / max(float(np.sum(areas)), 1e-8))
    return output


def evaluate_prompt(
    output_root: str,
    prompt: dict[str, str],
    variant: str,
    negative_strategy: str,
    lambda_neg: float,
    area_ratios: tuple[float, ...],
    smoothing_alphas: tuple[float, ...],
    baseline: str,
    seed: int,
) -> list[dict]:
    root = Path(output_root)
    image_id = prompt["image_id"]
    superpixel_dir = root / PRET_DIR / image_id
    records = read_csv(superpixel_dir / "superpixels.csv")
    tokens = np.load(superpixel_dir / f"tokens_{variant}.npy", mmap_mode="r")
    positive = parse_segment_ids(prompt["positive_segments"])
    negative = parse_segment_ids(prompt["negative_segments"])
    if not positive:
        return []
    labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records])
    areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
    target = labels == int(prompt["class_id"])
    token_values = np.asarray(tokens, dtype=np.float32)
    effective_positive = positive
    effective_negative = negative
    effective_variant = variant
    if baseline == "random_prompt":
        rng = np.random.default_rng(stable_seed(seed, prompt["query_id"], variant, prompt["shot"], prompt["prompt_source"]))
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < len(positive):
            return []
        effective_positive = rng.choice(valid_indices, size=len(positive), replace=False).astype(int).tolist()
        remaining = np.setdiff1d(valid_indices, np.asarray(effective_positive), assume_unique=False)
        if negative and len(remaining) >= len(negative):
            effective_negative = rng.choice(remaining, size=len(negative), replace=False).astype(int).tolist()
        else:
            effective_negative = []
        effective_variant = f"{variant}__random_prompt"
    elif baseline == "shuffled_token":
        rng = np.random.default_rng(stable_seed(seed, image_id, variant, "shuffled_token"))
        token_values = token_values[rng.permutation(len(token_values))]
        effective_variant = f"{variant}__shuffled_token"
    elif baseline != "none":
        raise ValueError(f"unknown baseline: {baseline}")
    raw_scores = scores_from_prompt(
        token_values, effective_positive, effective_negative, negative_strategy, lambda_neg
    )
    cache_key = (str(root), image_id)
    if cache_key not in NEIGHBOR_CACHE:
        segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
        NEIGHBOR_CACHE[cache_key] = segment_adjacency(np.asarray(segments), len(records))
    neighbors = NEIGHBOR_CACHE[cache_key]
    prompt_mask = np.zeros(len(records), dtype=bool)
    prompt_mask[effective_positive] = True
    if effective_negative:
        prompt_mask[effective_negative] = True
    outputs = []
    for alpha in smoothing_alphas:
        scores = smooth_scores(raw_scores, neighbors, alpha) if alpha > 0 else raw_scores
        for scope, candidate in [
            ("exclude_prompt_region", valid & ~prompt_mask),
            ("include_prompt_region", valid),
        ]:
            if int(np.sum(candidate & target)) == 0 or int(np.sum(candidate & ~target)) == 0:
                continue
            candidate_target = target[candidate]
            candidate_scores = scores[candidate]
            candidate_areas = areas[candidate]
            candidate_neighbors = candidate_neighbors_from_global(neighbors, candidate)
            metrics = ranking_metrics(candidate_target, candidate_scores, ks=(100, 1000))
            if not metrics.get("valid"):
                continue
            row = {
                "query_id": prompt["query_id"],
                "image_id": image_id,
                "class_id": int(prompt["class_id"]),
                "variant": effective_variant,
                "base_variant": variant,
                "baseline": baseline,
                "prompt_source": prompt["prompt_source"],
                "prompt_mode": prompt["prompt_mode"],
                "shot": int(prompt["shot"]),
                "negative_strategy": negative_strategy if effective_negative else "none",
                "smoothing_alpha": float(alpha),
                "scope": scope,
                "candidate_segments": int(candidate.sum()),
                "positive_segment_count": int(prompt["positive_segment_count"]),
                "negative_segment_count": int(prompt["negative_segment_count"]),
                "effective_positive_segment_count": len(effective_positive),
                "effective_negative_segment_count": len(effective_negative),
                "prompt_purity": float(prompt["prompt_purity"]),
                "prompt_target_area_fraction": float(prompt.get("prompt_target_area_fraction", 0.0)),
                "prompt_valid_area_fraction": float(prompt.get("prompt_valid_area_fraction", 0.0)),
                "prompt_area_10x_pixels": float(prompt.get("prompt_area_10x_pixels", 0.0)),
                "average_precision": metrics["average_precision"],
                "auroc": metrics["auroc"],
                "precision_at_100": metrics["precision_at_100"],
                "recall_at_1000": metrics["recall_at_1000"],
                "precision_at_top1_area": precision_at_area(candidate_target, candidate_scores, candidate_areas, 0.01),
                "precision_at_top5_area": precision_at_area(candidate_target, candidate_scores, candidate_areas, 0.05),
                "precision_at_top10_area": precision_at_area(candidate_target, candidate_scores, candidate_areas, 0.10),
            }
            if effective_negative and negative_strategy in {"negative_median", "negative_bank_max"}:
                pred_fixed = scores >= 0
                fixed = binary_segmentation_metrics(target[candidate], pred_fixed[candidate])
            else:
                pred_fixed = predict_top_area(candidate_scores, candidate_areas, 0.10)
                fixed = binary_segmentation_metrics(candidate_target, pred_fixed)
            row.update(
                {
                    "fixed_threshold_dice": fixed["dice"],
                    "fixed_threshold_iou": fixed["iou"],
                    "fixed_threshold_fp": fixed["false_positive"],
                    "fixed_threshold_fn": fixed["false_negative"],
                    **area_sweep_metrics(candidate_target, candidate_scores, candidate_areas, area_ratios),
                    **threshold_metrics(
                        candidate_target,
                        candidate_scores,
                        candidate_areas,
                        scores[np.asarray(effective_positive, dtype=np.int64)],
                        candidate_neighbors,
                    ),
                }
            )
            outputs.append(row)
    return outputs


def _job(
    output_root: str,
    prompts: list[dict[str, str]],
    variant: str,
    negative_strategy: str,
    lambda_neg: float,
    area_ratios: tuple[float, ...],
    smoothing_alphas: tuple[float, ...],
    baselines: tuple[str, ...],
    seed: int,
) -> tuple[str, list[dict]]:
    rows = []
    for prompt in prompts:
        for baseline in baselines:
            rows.extend(
                evaluate_prompt(
                    output_root,
                    prompt,
                    variant,
                    negative_strategy,
                    lambda_neg,
                    area_ratios,
                    smoothing_alphas,
                    baseline,
                    seed,
                )
            )
    return variant, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRET-style in-context superpixel segmentation.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--variants", nargs="+", default=["image_only", "cell_reg", "image_cell_reg"])
    parser.add_argument("--negative_strategy", choices=["negative_median", "negative_bank_max"], default="negative_bank_max")
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--area_ratios", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.18, 0.20, 0.30])
    parser.add_argument("--smoothing_alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--baselines", nargs="+", choices=["none", "random_prompt", "shuffled_token"], default=["none"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    prompts_csv = args.prompts_csv or str(Path(args.output_root) / PRET_DIR / "prompts.csv")
    prompts = read_csv(prompts_csv)
    if args.image_id:
        requested = set(args.image_id)
        prompts = [prompt for prompt in prompts if prompt["image_id"] in requested]
    prompts_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in prompts:
        prompts_by_image[prompt["image_id"]].append(prompt)
    ordered = [(variant, image_id) for variant in args.variants for image_id in prompts_by_image]
    results = []
    area_ratios = tuple(float(value) for value in args.area_ratios)
    smoothing_alphas = tuple(float(value) for value in args.smoothing_alphas)
    baselines = tuple(args.baselines)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _job,
                args.output_root,
                prompts_by_image[image_id],
                variant,
                args.negative_strategy,
                args.lambda_neg,
                area_ratios,
                smoothing_alphas,
                baselines,
                args.seed,
            ): (variant, image_id)
            for variant, image_id in ordered
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            variant, image_id = futures[future]
            _, rows = future.result()
            results.extend(rows)
            print(f"pret_eval {completed}/{len(futures)} variant={variant} image_id={image_id} rows={len(rows)}", flush=True)
    if not results:
        raise RuntimeError("PRET in-context evaluation produced no valid results")
    output_dir = Path(args.output_root) / PRET_DIR
    write_csv_atomic(output_dir / "pret_metrics.csv", results)
    summary = {}
    for key in sorted({(row["variant"], row["baseline"], float(row["smoothing_alpha"]), row["prompt_source"], row["scope"]) for row in results}):
        variant, baseline, smoothing_alpha, prompt_source, scope = key
        subset = [
            row for row in results
            if (
                row["variant"],
                row["baseline"],
                float(row["smoothing_alpha"]),
                row["prompt_source"],
                row["scope"],
            ) == key
        ]
        summary_key = f"{variant}|{baseline}|alpha={smoothing_alpha:g}|{prompt_source}|{scope}"
        summary[summary_key] = {
            "queries": len(subset),
            "mean_average_precision": float(np.mean([row["average_precision"] for row in subset])),
            "mean_auroc": float(np.mean([row["auroc"] for row in subset])),
            "mean_precision_at_top5_area": float(np.mean([row["precision_at_top5_area"] for row in subset])),
            "mean_fixed_threshold_dice": float(np.mean([row["fixed_threshold_dice"] for row in subset])),
            "mean_top1_area_dice": float(np.mean([row.get("top1_area_dice", 0.0) for row in subset])),
            "mean_top5_area_dice": float(np.mean([row.get("top5_area_dice", 0.0) for row in subset])),
            "mean_top10_area_dice": float(np.mean([row["top10_area_dice"] for row in subset])),
            "mean_top20_area_dice": float(np.mean([row.get("top20_area_dice", 0.0) for row in subset])),
            "mean_best_area_dice": float(np.mean([row.get("best_area_dice", 0.0) for row in subset])),
            "mean_best_area_ratio": float(np.mean([row.get("best_area_ratio", 0.0) for row in subset])),
            "mean_prompt_target_area_fraction": float(np.mean([row["prompt_target_area_fraction"] for row in subset])),
            "mean_prompt_valid_area_fraction": float(np.mean([row["prompt_valid_area_fraction"] for row in subset])),
        }
        for field in SUMMARY_EXTRA_FIELDS:
            summary[summary_key][f"mean_{field}"] = float(np.mean([row.get(field, 0.0) for row in subset]))
    write_json_atomic(output_dir / "pret_summary.json", summary)
    by_class = []
    for key in sorted({(row["variant"], row["baseline"], float(row["smoothing_alpha"]), row["prompt_source"], row["scope"], int(row["class_id"])) for row in results}):
        variant, baseline, smoothing_alpha, prompt_source, scope, class_id = key
        subset = [
            row for row in results
            if (
                row["variant"],
                row["baseline"],
                float(row["smoothing_alpha"]),
                row["prompt_source"],
                row["scope"],
                int(row["class_id"]),
            ) == key
        ]
        by_class.append(
            {
                "variant": variant,
                "baseline": baseline,
                "smoothing_alpha": smoothing_alpha,
                "prompt_source": prompt_source,
                "scope": scope,
                "class_id": class_id,
                "queries": len(subset),
                "mean_average_precision": float(np.mean([row["average_precision"] for row in subset])),
                "mean_precision_at_top5_area": float(np.mean([row["precision_at_top5_area"] for row in subset])),
                "mean_fixed_threshold_dice": float(np.mean([row["fixed_threshold_dice"] for row in subset])),
                "mean_percentile_90_dice": float(np.mean([row.get("percentile_90_dice", 0.0) for row in subset])),
                "mean_otsu_dice": float(np.mean([row.get("otsu_dice", 0.0) for row in subset])),
                "mean_cc_top20_keep3_dice": float(np.mean([row.get("cc_top20_keep3_dice", 0.0) for row in subset])),
                "mean_calibrated_top18_area_dice": float(np.mean([row.get("calibrated_top18_area_dice", 0.0) for row in subset])),
                "mean_best_area_dice": float(np.mean([row.get("best_area_dice", 0.0) for row in subset])),
                "acellular_focus": class_id in {3, 9, 10},
            }
        )
    write_json_atomic(output_dir / "pret_by_class.json", by_class)
    validation = {
        "passed": bool(results and all(np.isfinite(value) for row in results for value in row.values() if isinstance(value, (int, float)))),
        "results": len(results),
        "variants": args.variants,
        "baselines": list(baselines),
        "smoothing_alphas": list(smoothing_alphas),
        "area_ratios": list(area_ratios),
        "prompt_sources": sorted({row["prompt_source"] for row in results}),
        "scopes": sorted({row["scope"] for row in results}),
        "threshold_protocol": (
            "ranking metrics are threshold-free; deployable masks use only score distribution, "
            "prompt scores, segment areas, and HE-derived superpixel adjacency; calibrated top18/percentile90 "
            "are reported separately; BestDice is oracle over fixed area ratios"
        ),
        "gt_used_for_scoring": False,
    }
    write_json_atomic(output_dir / "pret_eval_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"PRET evaluation validation failed: {validation}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
