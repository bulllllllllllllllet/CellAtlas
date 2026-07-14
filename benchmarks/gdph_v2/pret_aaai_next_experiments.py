from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.gdph_v2.eval_retrieval import ranking_metrics
from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_aaai_eval import (
    CLASS_NAMES,
    _candidate_neighbors,
    _gmm_mask,
    _graph_boundary_f1,
)
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    area_sweep_metrics,
    binary_segmentation_metrics,
    normalized_median,
    otsu_score_threshold,
    parse_segment_ids,
    percentile_threshold,
    prompt_derived_class,
    precision_at_area,
    predict_top_area,
    read_csv,
    safe_l2_normalize,
    segment_adjacency,
    stable_seed,
    write_csv_atomic,
    write_json_atomic,
)


PRIMARY_VARIANT = "image_cell_reg_cellw0p5"
FRACTIONS = tuple(float(x) for x in np.linspace(0.01, 0.60, 60))
LEGACY_AREA_FEATURES = (
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "score_p80",
    "score_p90",
    "score_p95",
    "score_gap_top5_top20",
    "candidate_segments",
    "prompt_area_10x_pixels",
    "positive_prompt_count",
    "negative_prompt_count",
    "prototype_compactness",
    "top5_score_mean",
    "top10_score_mean",
    "top20_score_mean",
)
INTERACTION_PROTOCOLS = (
    "1pos",
    "2pos",
    "4pos",
    "8pos",
    "1pos_1neg",
    "1pos_2neg",
    "1pos_3neg",
    "1pos_5neg",
    "1pos_1strictneg",
    "1pos_2strictneg",
    "1pos_3strictneg",
    "2pos_1strictneg",
    "2pos_3strictneg",
    "4pos_1strictneg",
    "4pos_3strictneg",
    "8pos_3strictneg",
    "4pos_3neg",
    "1pos_1oracle_neg",
    "1pos_3oracle_neg",
)


def _dedupe_prompts(rows: list[dict[str, str]], prompt_source: str = "realistic_box") -> list[dict[str, str]]:
    seen: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("prompt_source") != prompt_source:
            continue
        seen.setdefault((row["query_id"], row["prompt_source"], int(row["shot"])), row)
    return [seen[key] for key in sorted(seen)]


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prototype(normalized_tokens: np.ndarray, segment_ids: list[int]) -> np.ndarray:
    if not segment_ids:
        return np.zeros(normalized_tokens.shape[1], dtype=np.float32)
    return normalized_median(normalized_tokens[np.asarray(segment_ids, dtype=np.int64)])


def _score_with_prompts(
    tokens: np.ndarray,
    positive_groups: list[list[int]],
    negative_groups: list[list[int]],
    lambda_neg: float,
) -> tuple[np.ndarray, int, int, float]:
    normalized = safe_l2_normalize(tokens, axis=1)
    pos_protos = np.stack([_prototype(normalized, group) for group in positive_groups], axis=0)
    pos_protos = safe_l2_normalize(pos_protos, axis=1)
    pos_scores = normalized @ pos_protos.T
    score = pos_scores.max(axis=1)
    compactness_values = []
    for proto, group in zip(pos_protos, positive_groups):
        if group:
            compactness_values.extend((normalized[np.asarray(group, dtype=np.int64)] @ proto).tolist())
    compactness = float(np.mean(compactness_values)) if compactness_values else 0.0
    if negative_groups:
        neg_protos = np.stack([_prototype(normalized, group) for group in negative_groups], axis=0)
        neg_protos = safe_l2_normalize(neg_protos, axis=1)
        neg_scores = normalized @ neg_protos.T
        score = score - float(lambda_neg) * neg_scores.max(axis=1)
    return score.astype(np.float32, copy=False), len(positive_groups), len(negative_groups), compactness


def _box_center(row: dict[str, str]) -> np.ndarray:
    x0 = _safe_float(row.get("x0_original"))
    y0 = _safe_float(row.get("y0_original"))
    x1 = _safe_float(row.get("x1_original"))
    y1 = _safe_float(row.get("y1_original"))
    return np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)


def _select_diverse_positive(
    base: dict[str, str],
    candidates: list[dict[str, str]],
    target_count: int,
) -> list[dict[str, str]]:
    selected = [base]
    used = {base["query_id"]}
    pool = [row for row in candidates if row["query_id"] not in used]
    while len(selected) < target_count and pool:
        selected_centers = np.stack([_box_center(row) for row in selected], axis=0)
        best_index = 0
        best_distance = -1.0
        for index, row in enumerate(pool):
            center = _box_center(row)
            distance = float(np.min(np.linalg.norm(selected_centers - center[None, :], axis=1)))
            if distance > best_distance:
                best_distance = distance
                best_index = index
        chosen = pool.pop(best_index)
        selected.append(chosen)
        used.add(chosen["query_id"])
    return selected


def _select_hard_negative(
    tokens: np.ndarray,
    positive_groups: list[list[int]],
    candidates: list[dict[str, str]],
    count: int,
    labels: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    purities: np.ndarray | None = None,
    target_class: int | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    if count <= 0 or not candidates:
        return []
    normalized = safe_l2_normalize(tokens, axis=1)
    pos_proto = _prototype(normalized, [sid for group in positive_groups for sid in group])
    scored = []
    for row in candidates:
        segs = parse_segment_ids(row.get("positive_segments", ""))
        if not segs:
            continue
        if strict:
            if labels is None or valid is None or purities is None or target_class is None:
                continue
            seg_array = np.asarray(segs, dtype=np.int64)
            if np.any(~valid[seg_array]):
                continue
            if np.any(labels[seg_array] == int(target_class)):
                continue
            if np.any(purities[seg_array] < 0.7):
                continue
        proto = _prototype(normalized, segs)
        scored.append((float(pos_proto @ proto), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:count]]


def _select_oracle_negative(
    tokens: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    target_class: int,
    positive_groups: list[list[int]],
    count: int,
) -> list[list[int]]:
    if count <= 0:
        return []
    normalized = safe_l2_normalize(tokens, axis=1)
    pos_proto = _prototype(normalized, [sid for group in positive_groups for sid in group])
    candidate = np.flatnonzero(valid & (labels != int(target_class)))
    if candidate.size == 0:
        return []
    scores = normalized[candidate] @ pos_proto
    order = np.argsort(-scores, kind="stable")
    return [[int(candidate[index])] for index in order[:count]]


def _interaction_groups(
    base: dict[str, str],
    by_image_class: dict[tuple[str, int], list[dict[str, str]]],
    by_image: dict[str, list[dict[str, str]]],
    tokens: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    purities: np.ndarray,
    protocol: str,
) -> tuple[list[list[int]], list[list[int]], str, str]:
    image_id = base["image_id"]
    class_id = int(base["class_id"])
    pos_count = 1
    neg_count = 0
    oracle_neg = False
    strict_neg = False
    if protocol == "2pos":
        pos_count = 2
    elif protocol == "4pos":
        pos_count = 4
    elif protocol == "8pos":
        pos_count = 8
    elif protocol == "1pos_1neg":
        neg_count = 1
    elif protocol == "1pos_2neg":
        neg_count = 2
    elif protocol == "1pos_3neg":
        neg_count = 3
    elif protocol == "1pos_5neg":
        neg_count = 5
    elif protocol == "1pos_1strictneg":
        neg_count = 1
        strict_neg = True
    elif protocol == "1pos_2strictneg":
        neg_count = 2
        strict_neg = True
    elif protocol == "1pos_3strictneg":
        neg_count = 3
        strict_neg = True
    elif protocol == "2pos_1strictneg":
        pos_count = 2
        neg_count = 1
        strict_neg = True
    elif protocol == "2pos_3strictneg":
        pos_count = 2
        neg_count = 3
        strict_neg = True
    elif protocol == "4pos_1strictneg":
        pos_count = 4
        neg_count = 1
        strict_neg = True
    elif protocol == "4pos_3strictneg":
        pos_count = 4
        neg_count = 3
        strict_neg = True
    elif protocol == "8pos_3strictneg":
        pos_count = 8
        neg_count = 3
        strict_neg = True
    elif protocol == "4pos_3neg":
        pos_count = 4
        neg_count = 3
    elif protocol == "1pos_1oracle_neg":
        neg_count = 1
        oracle_neg = True
    elif protocol == "1pos_3oracle_neg":
        neg_count = 3
        oracle_neg = True
    elif protocol != "1pos":
        raise ValueError(f"unknown interaction protocol: {protocol}")

    positives = _select_diverse_positive(base, by_image_class[(image_id, class_id)], pos_count)
    if len(positives) < pos_count:
        return [], [], "insufficient_positive", "none"
    positive_groups = [parse_segment_ids(row.get("positive_segments", "")) for row in positives]
    if any(not group for group in positive_groups):
        return [], [], "empty_positive", "none"

    negative_groups: list[list[int]] = []
    negative_source = "none"
    if neg_count and oracle_neg:
        negative_groups = _select_oracle_negative(tokens, labels, valid, class_id, positive_groups, neg_count)
        negative_source = "oracle_gt"
    elif neg_count:
        neg_candidates = [row for row in by_image[image_id] if int(row["class_id"]) != class_id]
        negatives = _select_hard_negative(
            tokens,
            positive_groups,
            neg_candidates,
            neg_count,
            labels=labels,
            valid=valid,
            purities=purities,
            target_class=class_id,
            strict=strict_neg,
        )
        negative_groups = [parse_segment_ids(row.get("positive_segments", "")) for row in negatives]
        negative_source = "realistic_hard_strict" if strict_neg else "realistic_hard"
    if neg_count and len(negative_groups) < neg_count:
        return [], [], "insufficient_negative", negative_source
    return positive_groups, negative_groups, "ok", negative_source


def _prompt_group_stats(
    groups: list[list[int]],
    labels: np.ndarray,
    purities: np.ndarray,
    valid: np.ndarray,
    target_class: int,
) -> dict[str, float | int | str]:
    segment_ids = [int(item) for group in groups for item in group]
    if not segment_ids:
        return {
            "segment_count": 0,
            "majority_label_mode": "",
            "target_fraction": 0.0,
            "mean_purity": 0.0,
            "min_purity": 0.0,
            "valid_fraction": 0.0,
        }
    idx = np.asarray(segment_ids, dtype=np.int64)
    group_labels = labels[idx]
    values, counts = np.unique(group_labels, return_counts=True)
    mode = int(values[int(np.argmax(counts))]) if len(values) else -1
    return {
        "segment_count": int(len(segment_ids)),
        "majority_label_mode": mode,
        "target_fraction": float(np.mean(group_labels == int(target_class))),
        "mean_purity": float(np.mean(purities[idx])),
        "min_purity": float(np.min(purities[idx])),
        "valid_fraction": float(np.mean(valid[idx])),
    }


def _best_area_ratio(target: np.ndarray, scores: np.ndarray, areas: np.ndarray) -> tuple[float, float]:
    sweep = area_sweep_metrics(target, scores, areas, FRACTIONS)
    return float(sweep["best_area_ratio"]), float(sweep["best_area_dice"])


def _candidate_features(
    scores: np.ndarray,
    areas: np.ndarray,
    prompt_area: float,
    positive_count: int,
    negative_count: int,
    compactness: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float64)
    top = {}
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        mask = predict_top_area(values, areas, frac)
        top[f"top{int(frac * 100)}_score_mean"] = float(np.mean(values[mask])) if np.any(mask) else 0.0
        top[f"top{int(frac * 100)}_score_std"] = float(np.std(values[mask])) if np.any(mask) else 0.0
    centered = finite - float(np.mean(finite))
    std = float(np.std(finite))
    skew = float(np.mean((centered / max(std, 1e-8)) ** 3)) if finite.size else 0.0
    return {
        "score_mean": float(np.mean(finite)),
        "score_std": std,
        "score_skew": skew,
        "score_min": float(np.min(finite)),
        "score_max": float(np.max(finite)),
        "score_range": float(np.max(finite) - np.min(finite)),
        "score_p80": float(np.percentile(finite, 80)),
        "score_p90": float(np.percentile(finite, 90)),
        "score_p95": float(np.percentile(finite, 95)),
        "score_p99": float(np.percentile(finite, 99)),
        "score_gap_p95_p80": float(np.percentile(finite, 95) - np.percentile(finite, 80)),
        "score_gap_top5_top20": top["top5_score_mean"] - top["top20_score_mean"],
        "score_gap_top1_top10": top["top1_score_mean"] - top["top10_score_mean"],
        "candidate_segments": float(len(values)),
        "candidate_area_10x_pixels": float(np.sum(areas)),
        "prompt_area_10x_pixels": float(prompt_area),
        "prompt_area_fraction_of_candidates": float(prompt_area / max(float(np.sum(areas)), 1e-8)),
        "positive_prompt_count": float(positive_count),
        "negative_prompt_count": float(negative_count),
        "prototype_compactness": float(compactness),
        **top,
    }


def _evaluate_image(
    source_root: str,
    prompts: list[dict[str, str]],
    all_prompts: list[dict[str, str]],
    protocols: tuple[str, ...],
    lambda_neg: float,
    primary_variant: str,
) -> tuple[str, list[dict]]:
    source = Path(source_root)
    image_id = prompts[0]["image_id"]
    superpixel_dir = source / PRET_DIR / image_id
    records = read_csv(superpixel_dir / "superpixels.csv")
    labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    purities = np.asarray([float(row["gt_label_purity"]) for row in records], dtype=np.float64)
    valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records])
    areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
    segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
    neighbors = segment_adjacency(np.asarray(segments), len(records))
    tokens = np.asarray(np.load(superpixel_dir / f"tokens_{primary_variant}.npy", mmap_mode="r"), dtype=np.float32)

    by_image_class: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_prompts:
        by_image[row["image_id"]].append(row)
        by_image_class[(row["image_id"], int(row["class_id"]))].append(row)

    rows: list[dict] = []
    for prompt in prompts:
        query_class_id = int(prompt["class_id"])
        for protocol in protocols:
            positive_groups, negative_groups, status, negative_source = _interaction_groups(
                prompt, by_image_class, by_image, tokens, labels, valid, purities, protocol
            )
            if status != "ok":
                continue
            class_id, eval_prompt_class_fraction = prompt_derived_class(
                positive_groups, labels, areas, default=query_class_id
            )
            target = labels == class_id
            positive_stats = _prompt_group_stats(positive_groups, labels, purities, valid, class_id)
            negative_stats = _prompt_group_stats(negative_groups, labels, purities, valid, class_id)
            prompt_mask = np.zeros(len(records), dtype=bool)
            for group in positive_groups:
                prompt_mask[np.asarray(group, dtype=np.int64)] = True
            scores, pos_used, neg_used, compactness = _score_with_prompts(tokens, positive_groups, negative_groups, lambda_neg)
            for scope, candidate in (
                ("exclude_prompt_region", valid & ~prompt_mask),
                ("include_prompt_region", valid),
            ):
                if int(np.sum(candidate & target)) == 0 or int(np.sum(candidate & ~target)) == 0:
                    continue
                candidate_scores = scores[candidate]
                candidate_target = target[candidate]
                candidate_areas = areas[candidate]
                ranking = ranking_metrics(candidate_target, candidate_scores, ks=(100, 1000))
                if not ranking.get("valid"):
                    continue
                best_area, best_dice = _best_area_ratio(candidate_target, candidate_scores, candidate_areas)
                feature_values = _candidate_features(
                    candidate_scores,
                    candidate_areas,
                    _safe_float(prompt.get("prompt_area_10x_pixels")),
                    pos_used,
                    neg_used,
                    compactness,
                )
                row = {
                    "query_id": prompt["query_id"],
                    "image_id": image_id,
                    "query_class_id": query_class_id,
                    "query_class_name": CLASS_NAMES[query_class_id] if 0 <= query_class_id < len(CLASS_NAMES) else str(query_class_id),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                    "eval_class_source": "prompt_area_weighted_majority",
                    "eval_prompt_class_fraction": eval_prompt_class_fraction,
                    "query_eval_class_match": int(query_class_id == class_id),
                    "scope": scope,
                    "variant": primary_variant,
                    "prompt_source": prompt.get("prompt_source", ""),
                    "shot": int(prompt.get("shot", 0)),
                    "interaction_protocol": protocol,
                    "positive_prompt_count": pos_used,
                    "negative_prompt_count": neg_used,
                    "negative_source": negative_source,
                    "lambda_neg": float(lambda_neg),
                    "candidate_segments": int(candidate.sum()),
                    "average_precision": float(ranking["average_precision"]),
                    "auroc": float(ranking["auroc"]),
                    "precision_at_top5_area": precision_at_area(candidate_target, candidate_scores, candidate_areas, 0.05),
                    "best_area_ratio": best_area,
                    "best_area_dice": best_dice,
                    "prompt_target_area_fraction": _safe_float(prompt.get("prompt_target_area_fraction")),
                    "prompt_purity": _safe_float(prompt.get("prompt_purity")),
                    "positive_gt_majority_label": positive_stats["majority_label_mode"],
                    "positive_gt_target_fraction": positive_stats["target_fraction"],
                    "positive_gt_mean_purity": positive_stats["mean_purity"],
                    "positive_gt_min_purity": positive_stats["min_purity"],
                    "negative_gt_majority_label": negative_stats["majority_label_mode"],
                    "negative_gt_target_fraction": negative_stats["target_fraction"],
                    "negative_gt_mean_purity": negative_stats["mean_purity"],
                    "negative_gt_min_purity": negative_stats["min_purity"],
                    "negative_gt_valid_fraction": negative_stats["valid_fraction"],
                    **feature_values,
                }
                rows.append(row)
    return image_id, rows


def _feature_columns(rows: list[dict]) -> list[str]:
    blocked = {
        "query_id",
        "image_id",
        "class_id",
        "class_name",
        "scope",
        "variant",
        "interaction_protocol",
        "negative_source",
        "average_precision",
        "auroc",
        "precision_at_top5_area",
        "best_area_ratio",
        "best_area_dice",
        "prompt_target_area_fraction",
        "prompt_purity",
        "positive_gt_majority_label",
        "positive_gt_target_fraction",
        "positive_gt_mean_purity",
        "positive_gt_min_purity",
        "negative_gt_majority_label",
        "negative_gt_target_fraction",
        "negative_gt_mean_purity",
        "negative_gt_min_purity",
        "negative_gt_valid_fraction",
    }
    numeric = []
    for key, value in rows[0].items():
        if key in blocked:
            continue
        if isinstance(value, (int, float)):
            numeric.append(key)
    return numeric


def _loio_area_predictions(
    rows: list[dict],
    seed: int,
    area_estimators: int,
    area_jobs: int,
) -> dict[tuple[str, str, str], dict[str, float]]:
    primary = [row for row in rows if row["scope"] == "exclude_prompt_region"]
    by_image = defaultdict(list)
    by_class_image = defaultdict(list)
    for row in primary:
        by_image[row["image_id"]].append(row)
        by_class_image[(int(row["class_id"]), row["image_id"])].append(row)
    feature_cols = _feature_columns(primary) if primary else []
    legacy_cols = [col for col in LEGACY_AREA_FEATURES if col in feature_cols]
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    all_images = sorted(by_image)
    global_fallback = float(np.median([float(row["best_area_ratio"]) for row in primary])) if primary else 0.18
    for fold_index, image_id in enumerate(all_images, start=1):
        train = [row for row in primary if row["image_id"] != image_id]
        if train:
            global_area = float(np.median([float(row["best_area_ratio"]) for row in train]))
        else:
            global_area = global_fallback
        class_train: dict[int, list[float]] = defaultdict(list)
        for row in train:
            class_train[int(row["class_id"])].append(float(row["best_area_ratio"]))
        legacy_model = None
        rf_v2_model = None
        linear_v2_model = None
        if len(train) >= 20 and legacy_cols:
            x_train = np.asarray([[float(row[col]) for col in legacy_cols] for row in train], dtype=np.float64)
            y_train = np.asarray([float(row["best_area_ratio"]) for row in train], dtype=np.float64)
            legacy_model = RandomForestRegressor(
                n_estimators=area_estimators,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=area_jobs,
            )
            legacy_model.fit(x_train, y_train)
        if len(train) >= 20 and feature_cols:
            x_train_v2 = np.asarray([[float(row[col]) for col in feature_cols] for row in train], dtype=np.float64)
            y_train = np.asarray([float(row["best_area_ratio"]) for row in train], dtype=np.float64)
            rf_v2_model = RandomForestRegressor(
                n_estimators=area_estimators,
                min_samples_leaf=5,
                random_state=seed + 17,
                n_jobs=area_jobs,
            )
            rf_v2_model.fit(x_train_v2, y_train)
            linear_v2_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            linear_v2_model.fit(x_train_v2, y_train)
        for row in by_image[image_id]:
            class_values = class_train.get(int(row["class_id"]), [])
            class_area = float(np.median(class_values)) if class_values else global_area
            if legacy_model is not None:
                x_legacy = np.asarray([[float(row[col]) for col in legacy_cols]], dtype=np.float64)
                prompt_area = float(legacy_model.predict(x_legacy)[0])
            else:
                prompt_area = global_area
            if rf_v2_model is not None:
                x_v2 = np.asarray([[float(row[col]) for col in feature_cols]], dtype=np.float64)
                prompt_area_v2 = float(rf_v2_model.predict(x_v2)[0])
            else:
                prompt_area_v2 = prompt_area
            if linear_v2_model is not None:
                linear_area = float(linear_v2_model.predict(x_v2)[0])
            else:
                linear_area = global_area
            prompt_area = float(np.clip(prompt_area, 0.01, 0.60))
            prompt_area_v2 = float(np.clip(prompt_area_v2, 0.01, 0.60))
            linear_area = float(np.clip(linear_area, 0.01, 0.60))
            key = (row["query_id"], row["interaction_protocol"], row["scope"])
            output[key] = {
                "global_toparea_loio": float(np.clip(global_area, 0.01, 0.60)),
                "classwise_toparea_loio": float(np.clip(class_area, 0.01, 0.60)),
                "prompt_adaptive_area_loio": prompt_area,
                "prompt_adaptive_area_loio_v2_rf": prompt_area_v2,
                "prompt_adaptive_area_loio_v2_linear": linear_area,
            }
        print(f"area_loio {fold_index}/{len(all_images)} image_id={image_id} train_rows={len(train)}", flush=True)
    return output


def _apply_thresholds(
    source_root: str,
    prompts: list[dict[str, str]],
    all_prompts: list[dict[str, str]],
    protocols: tuple[str, ...],
    area_predictions: dict[tuple[str, str, str], dict[str, float]],
    lambda_neg: float,
    seed: int,
    primary_variant: str,
) -> tuple[str, list[dict]]:
    source = Path(source_root)
    image_id = prompts[0]["image_id"]
    superpixel_dir = source / PRET_DIR / image_id
    records = read_csv(superpixel_dir / "superpixels.csv")
    labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    purities = np.asarray([float(row["gt_label_purity"]) for row in records], dtype=np.float64)
    valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records])
    areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
    segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
    neighbors = segment_adjacency(np.asarray(segments), len(records))
    tokens = np.asarray(np.load(superpixel_dir / f"tokens_{primary_variant}.npy", mmap_mode="r"), dtype=np.float32)

    by_image_class: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_prompts:
        by_image[row["image_id"]].append(row)
        by_image_class[(row["image_id"], int(row["class_id"]))].append(row)

    rows: list[dict] = []
    for prompt in prompts:
        query_class_id = int(prompt["class_id"])
        for protocol in protocols:
            positive_groups, negative_groups, status, negative_source = _interaction_groups(
                prompt, by_image_class, by_image, tokens, labels, valid, purities, protocol
            )
            if status != "ok":
                continue
            class_id, eval_prompt_class_fraction = prompt_derived_class(
                positive_groups, labels, areas, default=query_class_id
            )
            target = labels == class_id
            positive_stats = _prompt_group_stats(positive_groups, labels, purities, valid, class_id)
            negative_stats = _prompt_group_stats(negative_groups, labels, purities, valid, class_id)
            prompt_mask = np.zeros(len(records), dtype=bool)
            for group in positive_groups:
                prompt_mask[np.asarray(group, dtype=np.int64)] = True
            scores, pos_used, neg_used, compactness = _score_with_prompts(tokens, positive_groups, negative_groups, lambda_neg)
            for scope, candidate in (
                ("exclude_prompt_region", valid & ~prompt_mask),
                ("include_prompt_region", valid),
            ):
                if int(np.sum(candidate & target)) == 0 or int(np.sum(candidate & ~target)) == 0:
                    continue
                candidate_scores = scores[candidate]
                candidate_target = target[candidate]
                candidate_areas = areas[candidate]
                candidate_neighbors = _candidate_neighbors(neighbors, candidate)
                ranking = ranking_metrics(candidate_target, candidate_scores, ks=(100, 1000))
                if not ranking.get("valid"):
                    continue
                area_key = (prompt["query_id"], protocol, scope)
                areas_by_name = area_predictions.get(area_key, {})
                predictions = {
                    "p90": candidate_scores >= percentile_threshold(candidate_scores, 90.0),
                    "p80": candidate_scores >= percentile_threshold(candidate_scores, 80.0),
                    "otsu": candidate_scores >= otsu_score_threshold(candidate_scores),
                    "gmm_2comp": _gmm_mask(candidate_scores, stable_seed(seed, prompt["query_id"], protocol, "gmm")),
                }
                for name in (
                    "global_toparea_loio",
                    "classwise_toparea_loio",
                    "prompt_adaptive_area_loio",
                    "prompt_adaptive_area_loio_v2_rf",
                    "prompt_adaptive_area_loio_v2_linear",
                ):
                    if name in areas_by_name:
                        predictions[name] = predict_top_area(candidate_scores, candidate_areas, areas_by_name[name])
                for threshold_name, pred in predictions.items():
                    binary = binary_segmentation_metrics(candidate_target, pred)
                    rows.append(
                        {
                            "query_id": prompt["query_id"],
                            "image_id": image_id,
                            "query_class_id": query_class_id,
                            "query_class_name": CLASS_NAMES[query_class_id] if 0 <= query_class_id < len(CLASS_NAMES) else str(query_class_id),
                            "class_id": class_id,
                            "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                            "eval_class_source": "prompt_area_weighted_majority",
                            "eval_prompt_class_fraction": eval_prompt_class_fraction,
                            "query_eval_class_match": int(query_class_id == class_id),
                            "scope": scope,
                            "variant": primary_variant,
                            "prompt_source": prompt.get("prompt_source", ""),
                            "shot": int(prompt.get("shot", 0)),
                            "interaction_protocol": protocol,
                            "positive_prompt_count": pos_used,
                            "negative_prompt_count": neg_used,
                            "negative_source": negative_source,
                            "threshold_protocol": threshold_name,
                            "candidate_segments": int(candidate.sum()),
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
                            "best_area_ratio": float(area_predictions.get(area_key, {}).get("classwise_toparea_loio", 0.0)),
                            "prototype_compactness": float(compactness),
                            "prompt_target_area_fraction": _safe_float(prompt.get("prompt_target_area_fraction")),
                            "prompt_purity": _safe_float(prompt.get("prompt_purity")),
                            "positive_gt_majority_label": positive_stats["majority_label_mode"],
                            "positive_gt_target_fraction": positive_stats["target_fraction"],
                            "positive_gt_mean_purity": positive_stats["mean_purity"],
                            "positive_gt_min_purity": positive_stats["min_purity"],
                            "negative_gt_majority_label": negative_stats["majority_label_mode"],
                            "negative_gt_target_fraction": negative_stats["target_fraction"],
                            "negative_gt_mean_purity": negative_stats["mean_purity"],
                            "negative_gt_min_purity": negative_stats["min_purity"],
                            "negative_gt_valid_fraction": negative_stats["valid_fraction"],
                        }
                    )
    return image_id, rows


def _summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary = []
    by_class = []
    keys = sorted({(row["interaction_protocol"], row["threshold_protocol"], row["scope"], row["negative_source"]) for row in rows})
    for protocol, threshold, scope, negative_source in keys:
        subset = [
            row
            for row in rows
            if row["interaction_protocol"] == protocol
            and row["threshold_protocol"] == threshold
            and row["scope"] == scope
            and row["negative_source"] == negative_source
        ]
        summary.append(
            {
                "interaction_protocol": protocol,
                "negative_source": negative_source,
                "threshold_protocol": threshold,
                "scope": scope,
                "queries": len(subset),
                "mean_average_precision": _mean(subset, "average_precision"),
                "mean_auroc": _mean(subset, "auroc"),
                "mean_precision_at_top5_area": _mean(subset, "precision_at_top5_area"),
                "mean_dice": _mean(subset, "dice"),
                "mean_iou": _mean(subset, "iou"),
                "mean_binary_miou": _mean(subset, "binary_miou"),
                "mean_precision": _mean(subset, "precision"),
                "mean_recall": _mean(subset, "recall"),
                "mean_pred_area_fraction": _mean(subset, "pred_area_fraction"),
                "mean_boundary_f1_5px": _mean(subset, "boundary_f1_5px"),
                "mean_boundary_f1_10px": _mean(subset, "boundary_f1_10px"),
            }
        )
        for class_id in sorted({int(row["class_id"]) for row in subset}):
            class_subset = [row for row in subset if int(row["class_id"]) == class_id]
            by_class.append(
                {
                    "interaction_protocol": protocol,
                    "negative_source": negative_source,
                    "threshold_protocol": threshold,
                    "scope": scope,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                    "queries": len(class_subset),
                    "mean_average_precision": _mean(class_subset, "average_precision"),
                    "mean_dice": _mean(class_subset, "dice"),
                    "mean_iou": _mean(class_subset, "iou"),
                    "mean_precision": _mean(class_subset, "precision"),
                    "mean_recall": _mean(class_subset, "recall"),
                    "mean_boundary_f1_5px": _mean(class_subset, "boundary_f1_5px"),
                    "mean_boundary_f1_10px": _mean(class_subset, "boundary_f1_10px"),
                }
            )
    return summary, by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="AAAI next experiments: area calibration and multi-prompt interaction curves.")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--prompt_source", default="realistic_box")
    parser.add_argument("--protocols", nargs="+", default=list(INTERACTION_PROTOCOLS))
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--area_estimators", type=int, default=80)
    parser.add_argument("--primary_variant", default=PRIMARY_VARIANT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.source_root)
    output = Path(args.output_root)
    output_dir = output / PRET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts_csv) if args.prompts_csv else source / PRET_DIR / "prompts.csv"
    all_prompts = _dedupe_prompts(read_csv(prompts_path), args.prompt_source)
    if not all_prompts:
        raise RuntimeError(f"no prompts found for prompt_source={args.prompt_source}: {prompts_path}")
    if args.image_id:
        requested = set(args.image_id)
        all_prompts = [row for row in all_prompts if row["image_id"] in requested]
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in all_prompts:
        by_image[prompt["image_id"]].append(prompt)

    compact_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_image,
                str(source),
                image_prompts,
                all_prompts,
                tuple(args.protocols),
                args.lambda_neg,
                args.primary_variant,
            ): image_id
            for image_id, image_prompts in by_image.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id = futures[future]
            _, rows = future.result()
            compact_rows.extend(rows)
            print(f"next_compact {completed}/{len(futures)} image_id={image_id} rows={len(rows)}", flush=True)
    if not compact_rows:
        raise RuntimeError("no compact rows produced")
    write_csv_atomic(output_dir / "pret_aaai_next_compact.csv", compact_rows)
    area_predictions = _loio_area_predictions(compact_rows, args.seed, args.area_estimators, max(1, args.workers))

    metric_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _apply_thresholds,
                str(source),
                image_prompts,
                all_prompts,
                tuple(args.protocols),
                area_predictions,
                args.lambda_neg,
                args.seed,
                args.primary_variant,
            ): image_id
            for image_id, image_prompts in by_image.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id = futures[future]
            _, rows = future.result()
            metric_rows.extend(rows)
            print(f"next_metrics {completed}/{len(futures)} image_id={image_id} rows={len(rows)}", flush=True)
    if not metric_rows:
        raise RuntimeError("no metric rows produced")
    write_csv_atomic(output_dir / "pret_aaai_next_metrics.csv", metric_rows)
    summary, by_class_summary = _summaries(metric_rows)
    write_csv_atomic(output_dir / "pret_aaai_next_summary.csv", summary)
    write_csv_atomic(output_dir / "pret_aaai_next_by_class.csv", by_class_summary)
    write_json_atomic(
        output_dir / "pret_aaai_next_validation.json",
        {
            "passed": all(np.isfinite(float(value)) for row in metric_rows for value in row.values() if isinstance(value, (float, int))),
            "compact_rows": len(compact_rows),
            "metric_rows": len(metric_rows),
            "source_root": str(source),
            "output_root": str(output),
            "variant": PRIMARY_VARIANT,
            "primary_variant": args.primary_variant,
            "protocols": args.protocols,
            "threshold_protocols": sorted({row["threshold_protocol"] for row in metric_rows}),
            "note": "prompt-adaptive area protocols exclude GT-derived prompt purity/target fields and positive/negative GT audit fields from features.",
        },
    )
    print(json.dumps({"metrics": str(output_dir / "pret_aaai_next_metrics.csv"), "rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
