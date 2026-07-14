from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture

from benchmarks.gdph_v2.eval_retrieval import ranking_metrics
from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    binary_segmentation_metrics,
    connected_component_topk_mask,
    normalized_median,
    otsu_score_threshold,
    parse_segment_ids,
    percentile_threshold,
    precision_at_area,
    predict_top_area,
    read_csv,
    safe_l2_normalize,
    segment_adjacency,
    stable_seed,
    write_csv_atomic,
    write_json_atomic,
)


CLASS_NAMES = [
    "tumor_epithelium",
    "tumor_stroma",
    "background",
    "necrosis",
    "normal_gland",
    "normal_stroma",
    "submucosa_serosa",
    "muscle",
    "lymphocyte_aggregate",
    "mucus",
    "fat",
    "blood",
]


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _dedupe_prompts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        seen.setdefault((row["query_id"], row["prompt_source"], int(row["shot"])), row)
    return [seen[key] for key in sorted(seen)]


def _calibration_maps(source_root: Path, variant: str, prompt_source: str, scope: str) -> tuple[dict[tuple[int, str], float], float]:
    rows = [
        row
        for row in read_csv(source_root / PRET_DIR / "pret_metrics.csv")
        if row["variant"] == variant
        and row["baseline"] == "none"
        and row["prompt_source"] == prompt_source
        and row["scope"] == scope
        and float(row.get("smoothing_alpha", 0.0)) == 0.0
    ]
    by_class_image: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_class: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in rows:
        value = _float(row, "best_area_ratio")
        class_id = int(row["class_id"])
        image_id = row["image_id"]
        by_class_image[(class_id, image_id)].append(value)
        by_class[class_id].append(value)
        all_values.append(value)
    global_median = float(np.median(all_values)) if all_values else 0.18
    output: dict[tuple[int, str], float] = {}
    for class_id, values in by_class.items():
        for image_id in {key[1] for key in by_class_image if key[0] == class_id}:
            loio = []
            for (other_class, other_image), other_values in by_class_image.items():
                if other_class == class_id and other_image != image_id:
                    loio.extend(other_values)
            output[(class_id, image_id)] = float(np.median(loio)) if loio else float(np.median(values))
    return output, global_median


def _prototype_scores(tokens: np.ndarray, positive: list[int], protocol: str, seed: int) -> tuple[np.ndarray, int]:
    normalized = safe_l2_normalize(tokens, axis=1)
    pos_tokens = normalized[np.asarray(positive, dtype=np.int64)]
    if protocol == "median":
        prototypes = normalized_median(pos_tokens)[None, :]
    elif protocol.startswith("self_clean_drop"):
        drop = int(protocol.replace("self_clean_drop", ""))
        base = normalized_median(pos_tokens)
        keep_count = max(1, int(np.ceil(len(pos_tokens) * (1.0 - drop / 100.0))))
        similarities = pos_tokens @ base
        keep = np.argsort(similarities)[-keep_count:]
        prototypes = normalized_median(pos_tokens[keep])[None, :]
    elif protocol.startswith("multi_proto_k"):
        k = int(protocol.replace("multi_proto_k", ""))
        k = max(1, min(k, len(pos_tokens)))
        if k == 1:
            prototypes = normalized_median(pos_tokens)[None, :]
        else:
            base = normalized_median(pos_tokens)
            order = np.argsort(pos_tokens @ base, kind="stable")
            chunks = [chunk for chunk in np.array_split(order, k) if len(chunk)]
            proto_items = [normalized_median(pos_tokens[chunk]) for chunk in chunks]
            prototypes = np.stack(proto_items, axis=0) if proto_items else normalized_median(pos_tokens)[None, :]
    else:
        raise ValueError(f"unknown prototype protocol: {protocol}")
    scores = normalized @ safe_l2_normalize(prototypes, axis=1).T
    return scores.max(axis=1), int(len(prototypes))


def _gmm_mask(scores: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.zeros(len(values), dtype=bool)
    clean = values[finite]
    if len(clean) < 4 or float(clean.min()) == float(clean.max()):
        output[finite] = clean >= percentile_threshold(clean, 90.0)
        return output
    try:
        model = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
        labels = model.fit_predict(clean.reshape(-1, 1))
    except ValueError:
        output[finite] = clean >= percentile_threshold(clean, 90.0)
        return output
    positive_component = int(np.argmax(model.means_.reshape(-1)))
    output[finite] = labels == positive_component
    return output


def _boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int) -> float:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if not np.any(truth) and not np.any(pred):
        return 1.0
    if not np.any(truth) or not np.any(pred):
        return 0.0
    truth_boundary = truth ^ ndi.binary_erosion(truth)
    pred_boundary = pred ^ ndi.binary_erosion(pred)
    if not np.any(truth_boundary) or not np.any(pred_boundary):
        return 0.0
    truth_dist = ndi.distance_transform_edt(~truth_boundary)
    pred_dist = ndi.distance_transform_edt(~pred_boundary)
    precision = float(np.mean(truth_dist[pred_boundary] <= tolerance))
    recall = float(np.mean(pred_dist[truth_boundary] <= tolerance))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _expand_graph_boundary(boundary: np.ndarray, neighbors: list[np.ndarray], hops: int) -> np.ndarray:
    expanded = np.asarray(boundary, dtype=bool).copy()
    frontier = expanded.copy()
    for _ in range(hops):
        next_frontier = np.zeros_like(expanded)
        for index in np.flatnonzero(frontier):
            adjacent = neighbors[int(index)]
            if len(adjacent):
                next_frontier[adjacent] = True
        frontier = next_frontier & ~expanded
        expanded |= next_frontier
    return expanded


def _graph_boundary_f1(target: np.ndarray, pred: np.ndarray, neighbors: list[np.ndarray], hops: int) -> float:
    target = np.asarray(target, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    target_boundary = np.zeros(len(target), dtype=bool)
    pred_boundary = np.zeros(len(pred), dtype=bool)
    for index, adjacent in enumerate(neighbors):
        if len(adjacent) == 0:
            continue
        target_boundary[index] = bool(np.any(target[adjacent] != target[index]))
        pred_boundary[index] = bool(np.any(pred[adjacent] != pred[index]))
    if not np.any(target_boundary) and not np.any(pred_boundary):
        return 1.0
    if not np.any(target_boundary) or not np.any(pred_boundary):
        return 0.0
    target_dilated = _expand_graph_boundary(target_boundary, neighbors, hops)
    pred_dilated = _expand_graph_boundary(pred_boundary, neighbors, hops)
    precision = float(np.sum(pred_boundary & target_dilated) / max(np.sum(pred_boundary), 1))
    recall = float(np.sum(target_boundary & pred_dilated) / max(np.sum(target_boundary), 1))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _candidate_neighbors(global_neighbors: list[np.ndarray], candidate: np.ndarray) -> list[np.ndarray]:
    global_indices = np.flatnonzero(candidate)
    local_index = {int(value): idx for idx, value in enumerate(global_indices)}
    output = []
    for global_id in global_indices:
        output.append(np.asarray([local_index[int(n)] for n in global_neighbors[int(global_id)] if int(n) in local_index], dtype=np.int64))
    return output


def _threshold_predictions(
    scores: np.ndarray,
    areas: np.ndarray,
    neighbors: list[np.ndarray],
    calibration_area: float,
    seed: int,
) -> dict[str, np.ndarray]:
    output = {
        "otsu": scores >= otsu_score_threshold(scores),
        "gmm_2comp": _gmm_mask(scores, seed),
        "p80": scores >= percentile_threshold(scores, 80.0),
        "p85": scores >= percentile_threshold(scores, 85.0),
        "p90": scores >= percentile_threshold(scores, 90.0),
        "p95": scores >= percentile_threshold(scores, 95.0),
        "classwise_toparea_loio": predict_top_area(scores, areas, calibration_area),
        "cc_filter_top20_keep1": connected_component_topk_mask(scores, areas, neighbors, 0.20, 1) if neighbors else predict_top_area(scores, areas, 0.20),
        "cc_filter_top20_keep3": connected_component_topk_mask(scores, areas, neighbors, 0.20, 3) if neighbors else predict_top_area(scores, areas, 0.20),
        "cc_filter_top20_keep5": connected_component_topk_mask(scores, areas, neighbors, 0.20, 5) if neighbors else predict_top_area(scores, areas, 0.20),
    }
    return output


def _evaluate_image(
    output_root: str,
    source_root: str,
    prompts: list[dict[str, str]],
    variants: tuple[str, ...],
    prototype_protocols: tuple[str, ...],
    calibration_by_class_image: dict[tuple[int, str], float],
    global_calibration: float,
    seed: int,
) -> tuple[str, list[dict]]:
    source = Path(source_root)
    image_id = prompts[0]["image_id"]
    superpixel_dir = source / PRET_DIR / image_id
    records = read_csv(superpixel_dir / "superpixels.csv")
    labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records])
    areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
    segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
    neighbors = segment_adjacency(np.asarray(segments), len(records))
    tokens_by_variant = {
        variant: np.asarray(np.load(superpixel_dir / f"tokens_{variant}.npy", mmap_mode="r"), dtype=np.float32)
        for variant in variants
    }
    rows: list[dict] = []
    for prompt in prompts:
        if prompt["prompt_source"] != "realistic_box":
            continue
        positive = parse_segment_ids(prompt["positive_segments"])
        if not positive:
            continue
        prompt_mask = np.zeros(len(records), dtype=bool)
        prompt_mask[positive] = True
        class_id = int(prompt["class_id"])
        target = labels == class_id
        for scope, candidate in [
            ("exclude_prompt_region", valid & ~prompt_mask),
            ("include_prompt_region", valid),
        ]:
            if int(np.sum(candidate & target)) == 0 or int(np.sum(candidate & ~target)) == 0:
                continue
            candidate_target = target[candidate]
            candidate_areas = areas[candidate]
            candidate_neighbors = _candidate_neighbors(neighbors, candidate)
            calibration = calibration_by_class_image.get((class_id, image_id), global_calibration)
            for variant, tokens in tokens_by_variant.items():
                for prototype_protocol in prototype_protocols:
                    scores, proto_count = _prototype_scores(
                        tokens,
                        positive,
                        prototype_protocol,
                        stable_seed(seed, prompt["query_id"], variant, prototype_protocol),
                    )
                    candidate_scores = scores[candidate]
                    ranking = ranking_metrics(candidate_target, candidate_scores, ks=(100, 1000))
                    if not ranking.get("valid"):
                        continue
                    predictions = _threshold_predictions(
                        candidate_scores,
                        candidate_areas,
                        candidate_neighbors,
                        calibration,
                        stable_seed(seed, prompt["query_id"], "threshold"),
                    )
                    for threshold_name, pred in predictions.items():
                        binary = binary_segmentation_metrics(candidate_target, pred)
                        rows.append(
                            {
                                "query_id": prompt["query_id"],
                                "image_id": image_id,
                                "class_id": class_id,
                                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                                "shot": int(prompt["shot"]),
                                "prompt_source": prompt["prompt_source"],
                                "scope": scope,
                                "variant": variant,
                                "prototype_protocol": prototype_protocol,
                                "prototype_count": proto_count,
                                "threshold_protocol": threshold_name,
                                "candidate_segments": int(candidate.sum()),
                                "prompt_target_area_fraction": float(prompt.get("prompt_target_area_fraction", 0.0)),
                                "average_precision": float(ranking["average_precision"]),
                                "auroc": float(ranking["auroc"]),
                                "precision_at_top5_area": precision_at_area(candidate_target, candidate_scores, candidate_areas, 0.05),
                                "dice": float(binary["dice"]),
                                "iou": float(binary["iou"]),
                                "binary_miou": float(binary["iou"]),
                                "precision": float(binary["precision"]),
                                "recall": float(binary["recall"]),
                                "false_positive": int(binary["false_positive"]),
                                "false_negative": int(binary["false_negative"]),
                                "pred_area_fraction": float(np.sum(candidate_areas[pred]) / max(float(np.sum(candidate_areas)), 1e-8)),
                                "boundary_f1_5px": _graph_boundary_f1(candidate_target, pred, candidate_neighbors, 1),
                                "boundary_f1_10px": _graph_boundary_f1(candidate_target, pred, candidate_neighbors, 2),
                            }
                        )
    return image_id, rows


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else 0.0


def _summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary = []
    by_class = []
    summary_keys = sorted({(row["variant"], row["prototype_protocol"], row["threshold_protocol"], row["scope"]) for row in rows})
    for key in summary_keys:
        subset = [row for row in rows if (row["variant"], row["prototype_protocol"], row["threshold_protocol"], row["scope"]) == key]
        variant, prototype_protocol, threshold_protocol, scope = key
        summary.append(
            {
                "variant": variant,
                "prototype_protocol": prototype_protocol,
                "threshold_protocol": threshold_protocol,
                "scope": scope,
                "queries": len(subset),
                "mean_average_precision": _mean(subset, "average_precision"),
                "mean_auroc": _mean(subset, "auroc"),
                "mean_dice": _mean(subset, "dice"),
                "mean_iou": _mean(subset, "iou"),
                "mean_binary_miou": _mean(subset, "binary_miou"),
                "mean_boundary_f1_5px": _mean(subset, "boundary_f1_5px"),
                "mean_boundary_f1_10px": _mean(subset, "boundary_f1_10px"),
                "mean_pred_area_fraction": _mean(subset, "pred_area_fraction"),
            }
        )
        for class_id in sorted({int(row["class_id"]) for row in subset}):
            class_subset = [row for row in subset if int(row["class_id"]) == class_id]
            by_class.append(
                {
                    "variant": variant,
                    "prototype_protocol": prototype_protocol,
                    "threshold_protocol": threshold_protocol,
                    "scope": scope,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                    "queries": len(class_subset),
                    "mean_average_precision": _mean(class_subset, "average_precision"),
                    "mean_dice": _mean(class_subset, "dice"),
                    "mean_iou": _mean(class_subset, "iou"),
                    "mean_boundary_f1_5px": _mean(class_subset, "boundary_f1_5px"),
                    "mean_boundary_f1_10px": _mean(class_subset, "boundary_f1_10px"),
                }
            )
    return summary, by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="AAAI PRET-superpixel baselines and calibration evaluation.")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--variants", nargs="+", default=["image_cell_reg_cellw0p5", "patch_only", "image_only", "cell_reg"])
    parser.add_argument("--prototype_protocols", nargs="+", default=["median", "self_clean_drop10", "self_clean_drop20", "self_clean_drop30", "multi_proto_k2", "multi_proto_k3"])
    parser.add_argument("--calibration_variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.source_root)
    output = Path(args.output_root)
    output_dir = output / PRET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts_csv) if args.prompts_csv else source / PRET_DIR / "prompts.csv"
    prompts = _dedupe_prompts([row for row in read_csv(prompts_path) if row["prompt_source"] == "realistic_box"])
    if args.image_id:
        requested = set(args.image_id)
        prompts = [row for row in prompts if row["image_id"] in requested]
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in prompts:
        by_image[prompt["image_id"]].append(prompt)

    calibration, global_calibration = _calibration_maps(source, args.calibration_variant, "realistic_box", "exclude_prompt_region")
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_image,
                str(output),
                str(source),
                image_prompts,
                tuple(args.variants),
                tuple(args.prototype_protocols),
                calibration,
                global_calibration,
                args.seed,
            ): image_id
            for image_id, image_prompts in by_image.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id = futures[future]
            _, rows = future.result()
            results.extend(rows)
            print(f"pret_aaai_eval {completed}/{len(futures)} image_id={image_id} rows={len(rows)}", flush=True)
    if not results:
        raise RuntimeError("AAAI PRET evaluation produced no rows")
    write_csv_atomic(output_dir / "pret_aaai_metrics.csv", results)
    summary, by_class = _summaries(results)
    write_csv_atomic(output_dir / "pret_aaai_summary.csv", summary)
    write_csv_atomic(output_dir / "pret_aaai_by_class.csv", by_class)
    write_json_atomic(
        output_dir / "pret_aaai_validation.json",
        {
            "passed": all(np.isfinite(float(value)) for row in results for value in row.values() if isinstance(value, (float, int))),
            "results": len(results),
            "source_root": str(source),
            "variants": args.variants,
            "prototype_protocols": args.prototype_protocols,
            "threshold_protocols": sorted({row["threshold_protocol"] for row in results}),
            "global_classwise_toparea_fallback": global_calibration,
            "metric_note": "mIoU is binary target-vs-rest IoU averaged across prompt queries/classes, not a 12-class dense decoder mIoU.",
        },
    )
    print(json.dumps({"metrics": str(output_dir / "pret_aaai_metrics.csv"), "rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()
