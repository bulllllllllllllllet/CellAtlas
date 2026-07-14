from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


NUM_CLASSES = 12
PRET_DIR = "pret_superpixel"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_atomic(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_segment_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.split(";") if item != ""]


def format_segment_ids(values: list[int] | np.ndarray) -> str:
    return ";".join(str(int(value)) for value in values)


def original_to_target_scale(
    original_size: tuple[int, int], target_shape: tuple[int, int]
) -> tuple[float, float]:
    original_width, original_height = original_size
    target_height, target_width = target_shape
    return target_width / original_width, target_height / original_height


def box_original_to_target(
    box: tuple[float, float, float, float],
    original_size: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx, sy = original_to_target_scale(original_size, target_shape)
    target_height, target_width = target_shape
    x0, y0, x1, y1 = box
    tx0 = max(0, min(target_width, int(np.floor(x0 * sx))))
    ty0 = max(0, min(target_height, int(np.floor(y0 * sy))))
    tx1 = max(tx0 + 1, min(target_width, int(np.ceil(x1 * sx))))
    ty1 = max(ty0 + 1, min(target_height, int(np.ceil(y1 * sy))))
    return tx0, ty0, tx1, ty1


def points_original_to_target(
    xy: np.ndarray, original_size: tuple[int, int], target_shape: tuple[int, int]
) -> np.ndarray:
    sx, sy = original_to_target_scale(original_size, target_shape)
    points = np.asarray(xy, dtype=np.float64).copy()
    points[:, 0] *= sx
    points[:, 1] *= sy
    return points


def safe_l2_normalize(values: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, eps)


def normalized_median(values: np.ndarray) -> np.ndarray:
    normalized = safe_l2_normalize(np.asarray(values, dtype=np.float32), axis=1)
    prototype = np.median(normalized, axis=0)
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-8:
        return np.zeros_like(prototype, dtype=np.float32)
    return (prototype / norm).astype(np.float32, copy=False)


def prompt_derived_class(
    segment_groups: list[list[int]],
    labels: np.ndarray,
    areas: np.ndarray | None = None,
    default: int = 255,
    num_classes: int = NUM_CLASSES,
) -> tuple[int, float]:
    segment_ids = [
        int(segment_id)
        for group in segment_groups
        for segment_id in group
        if 0 <= int(segment_id) < len(labels)
    ]
    if not segment_ids:
        return int(default), 0.0
    prompt_labels = np.asarray(labels, dtype=np.int64)[np.asarray(segment_ids, dtype=np.int64)]
    valid = (prompt_labels >= 0) & (prompt_labels < num_classes)
    if not np.any(valid):
        return int(default), 0.0
    valid_labels = prompt_labels[valid].astype(np.int64)
    if areas is None:
        weights = np.ones(valid_labels.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(areas, dtype=np.float64)[np.asarray(segment_ids, dtype=np.int64)[valid]]
    counts = np.bincount(valid_labels, weights=weights, minlength=num_classes)
    class_id = int(np.argmax(counts))
    total = float(np.sum(counts))
    support = float(counts[class_id] / total) if total > 0 else 0.0
    return class_id, support


def weighted_concat(blocks: list[tuple[np.ndarray, float]]) -> np.ndarray:
    parts = []
    for block, weight in blocks:
        normalized = safe_l2_normalize(block, axis=1)
        parts.append(normalized * np.float32(weight))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def majority_label(values: np.ndarray, num_classes: int = NUM_CLASSES) -> tuple[int, float, float]:
    flat = np.asarray(values).reshape(-1)
    valid = flat[(flat >= 0) & (flat < num_classes)]
    if valid.size == 0:
        return 255, 0.0, 0.0
    counts = np.bincount(valid.astype(np.int64), minlength=num_classes)
    label = int(np.argmax(counts))
    return label, float(counts[label] / valid.size), float(valid.size / flat.size)


def binary_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "predicted_positive_fraction": float(np.mean(pred)) if pred.size else 0.0,
    }


def precision_at_area(target: np.ndarray, scores: np.ndarray, areas: np.ndarray, fraction: float) -> float:
    target = np.asarray(target, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    total_area = float(np.sum(areas))
    if total_area <= 0:
        return 0.0
    limit = total_area * fraction
    chosen = np.zeros(len(scores), dtype=bool)
    cumulative = 0.0
    for index in order:
        if cumulative >= limit:
            break
        chosen[index] = True
        cumulative += float(areas[index])
    selected_area = float(np.sum(areas[chosen]))
    return float(np.sum(areas[chosen & target]) / selected_area) if selected_area > 0 else 0.0


def predict_top_area(scores: np.ndarray, areas: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    selected = np.zeros(len(scores), dtype=bool)
    limit = float(np.sum(areas)) * fraction
    cumulative = 0.0
    for index in order:
        if cumulative >= limit:
            break
        selected[index] = True
        cumulative += float(areas[index])
    return selected


def otsu_score_threshold(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if float(values.min()) == float(values.max()):
        return float(values.max())
    hist, edges = np.histogram(values, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(hist).astype(np.float64)
    weight_right = np.cumsum(hist[::-1]).astype(np.float64)[::-1]
    sum_left = np.cumsum(hist * centers)
    sum_right = np.cumsum((hist * centers)[::-1])[::-1]
    valid = (weight_left > 0) & (weight_right > 0)
    mean_left = np.divide(sum_left, weight_left, out=np.zeros_like(sum_left), where=weight_left > 0)
    mean_right = np.divide(sum_right, weight_right, out=np.zeros_like(sum_right), where=weight_right > 0)
    between = weight_left * weight_right * (mean_left - mean_right) ** 2
    if not np.any(valid):
        return float(np.quantile(values, 0.9))
    return float(centers[int(np.argmax(np.where(valid, between, -1)))])


def mean_std_threshold(scores: np.ndarray, k: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean() + k * values.std())


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def prompt_relative_threshold(
    candidate_scores: np.ndarray,
    positive_scores: np.ndarray,
    margin: float,
    fallback_percentile: float = 90.0,
) -> float:
    candidates = np.asarray(candidate_scores, dtype=np.float64)
    candidates = candidates[np.isfinite(candidates)]
    positives = np.asarray(positive_scores, dtype=np.float64)
    positives = positives[np.isfinite(positives)]
    if candidates.size == 0:
        return 0.0
    if positives.size <= 1:
        return percentile_threshold(candidates, fallback_percentile)
    threshold = float(np.median(positives) - margin)
    lower = percentile_threshold(candidates, 50.0)
    upper = percentile_threshold(candidates, 98.0)
    return float(np.clip(threshold, lower, upper))


def connected_component_topk_mask(
    scores: np.ndarray,
    areas: np.ndarray,
    neighbors: list[np.ndarray],
    seed_fraction: float,
    max_components: int,
) -> np.ndarray:
    seed = predict_top_area(scores, areas, seed_fraction)
    if max_components <= 0 or not np.any(seed):
        return seed
    scores = np.asarray(scores, dtype=np.float64)
    seed_indices = set(np.flatnonzero(seed).astype(int).tolist())
    components: list[list[int]] = []
    while seed_indices:
        start = seed_indices.pop()
        stack = [start]
        component = [start]
        while stack:
            index = stack.pop()
            for adjacent in neighbors[index]:
                item = int(adjacent)
                if item in seed_indices:
                    seed_indices.remove(item)
                    stack.append(item)
                    component.append(item)
        components.append(component)
    components.sort(key=lambda items: float(np.mean(scores[items])) if items else -np.inf, reverse=True)
    selected = np.zeros(len(scores), dtype=bool)
    for component in components[:max_components]:
        selected[np.asarray(component, dtype=np.int64)] = True
    return selected


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def area_sweep_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    fractions: tuple[float, ...],
) -> dict[str, float]:
    output: dict[str, float] = {}
    best_dice = -1.0
    best_iou = 0.0
    best_fraction = 0.0
    for fraction in fractions:
        pred = predict_top_area(scores, areas, fraction)
        metrics = binary_segmentation_metrics(target, pred)
        label = f"top{int(round(fraction * 100))}_area"
        output[f"{label}_dice"] = float(metrics["dice"])
        output[f"{label}_iou"] = float(metrics["iou"])
        if float(metrics["dice"]) > best_dice:
            best_dice = float(metrics["dice"])
            best_iou = float(metrics["iou"])
            best_fraction = float(fraction)
    output["best_area_dice"] = best_dice if best_dice >= 0 else 0.0
    output["best_area_iou"] = best_iou
    output["best_area_ratio"] = best_fraction
    return output


def prompt_purity_bin(value: float) -> str:
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.9:
        return "0.7-0.9"
    return ">0.9"


def segment_adjacency(segments: np.ndarray, num_segments: int | None = None) -> list[np.ndarray]:
    values = np.asarray(segments)
    if num_segments is None:
        num_segments = int(values.max()) + 1 if np.any(values >= 0) else 0
    neighbors: list[set[int]] = [set() for _ in range(num_segments)]
    for a, b in (
        (values[:, :-1], values[:, 1:]),
        (values[:-1, :], values[1:, :]),
    ):
        mask = (a >= 0) & (b >= 0) & (a != b)
        for left, right in zip(a[mask].reshape(-1), b[mask].reshape(-1)):
            i = int(left)
            j = int(right)
            neighbors[i].add(j)
            neighbors[j].add(i)
    return [np.asarray(sorted(items), dtype=np.int64) for items in neighbors]


def smooth_scores(scores: np.ndarray, neighbors: list[np.ndarray], alpha: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if alpha <= 0:
        return values.copy()
    smoothed = values.copy()
    for index, adjacent in enumerate(neighbors):
        if len(adjacent):
            smoothed[index] = (1 - alpha) * values[index] + alpha * float(np.mean(values[adjacent]))
    return smoothed
