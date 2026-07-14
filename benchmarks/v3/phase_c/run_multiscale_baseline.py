# 作用：为 Phase B prompt task 在 small/medium/large 三个尺度上计算 baseline retrieval 分数与指标。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import (
    binary_segmentation_metrics,
    otsu_score_threshold,
    percentile_threshold,
    predict_top_area,
    safe_l2_normalize,
    segment_adjacency,
    stable_seed,
)


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST_PATH = PRET_ROOT / "data_manifest_v3.csv"
PROMPT_PATH = PRET_ROOT / "prompt_tasks" / "all_prompt_tasks.csv"
MULTISCALE_ROOT = PRET_ROOT / "multiscale_tokens"
EVAL_DIR = PRET_ROOT / "evaluations"
SCORE_DIR = EVAL_DIR / "query_scale_scores"
METRICS_PATH = EVAL_DIR / "multiscale_baseline_metrics.csv"
REPORTS_DIR = PRET_ROOT / "reports"
VALIDATION_PATH = REPORTS_DIR / "phase_c_validation.json"
PHASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = PHASE_DIR / "report.md"
PALETTE_JSON = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/gdph_tissue_palette.json")

SCALES = ("small", "medium", "large")
NUM_CLASSES = 12
FRACTIONS = tuple(float(x) for x in np.linspace(0.01, 0.60, 60))
DEFAULT_AREA_FLOOR = 0.01
CLASS_SCALE_AREA_FLOORS = {
    (11, "small"): 0.001,
    (11, "large"): 0.001,
}

METRIC_FIELDS = [
    "query_id",
    "image_id",
    "target_class",
    "scale",
    "prompt_quality",
    "prompt_mode",
    "positive_prompt_segments",
    "negative_prompt_segments",
    "candidate_segments",
    "mAP",
    "AUROC",
    "P@top5",
    "P@top10",
    "Dice_p90",
    "Dice_p80",
    "Dice_otsu",
    "Dice_gmm",
    "Dice_global_toparea",
    "Dice_classwise_toparea",
    "BestDice",
    "BestArea",
    "mIoU",
    "BF1@5",
    "BF1@10",
    "PredArea",
    "GTArea",
    "Precision",
    "Recall",
    "FP_area",
    "FN_area",
    "score_mean",
    "score_std",
    "score_p90",
    "gt_positive_segments",
    "gt_negative_segments",
    "status",
]


@dataclass
class ScaleData:
    image_id: str
    scale: str
    segment_ids: np.ndarray
    tokens: np.ndarray
    tokens_norm: np.ndarray
    areas: np.ndarray
    centers: np.ndarray
    boxes: np.ndarray
    valid: np.ndarray
    gt_majority_label: np.ndarray
    gt_purity: np.ndarray
    valid_fraction: np.ndarray
    adjacency: list[np.ndarray]
    edges: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_ids(value: object) -> list[int]:
    text = str(value or "")
    if not text:
        return []
    return [int(part) for part in text.split(";") if part != ""]


def parse_boxes(value: object) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    text = str(value or "")
    for part in text.split(";"):
        if not part:
            continue
        x0, y0, x1, y1 = [int(float(item)) for item in part.split(":")]
        boxes.append((x0, y0, x1, y1))
    return boxes


def load_gt_mask(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        palette = load_palette(PALETTE_JSON)
        return rgb_to_class_mask(array.astype(np.uint8), palette)
    return array.astype(np.int16, copy=False)


def load_palette(path: Path) -> np.ndarray:
    colors = np.zeros((256, 3), dtype=np.uint8)
    colors[:] = [35, 35, 35]
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("classes", []):
        colors[int(item["id"])] = item["rgb"]
    colors[255] = [0, 0, 0]
    return colors


def rgb_to_class_mask(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    output = np.full(rgb.shape[:2], 255, dtype=np.int16)
    for class_id, color in enumerate(palette[:NUM_CLASSES]):
        output[np.all(rgb == color.reshape(1, 1, 3), axis=2)] = class_id
    return output


def load_superpixel_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    segment_ids = np.asarray([int(float(row["segment_id"])) for row in rows], dtype=np.int64)
    areas = np.asarray([float(row.get("area", 0.0)) for row in rows], dtype=np.float64)
    centers = np.asarray(
        [[float(row.get("center_x", 0.0)), float(row.get("center_y", 0.0))] for row in rows],
        dtype=np.float64,
    )
    boxes = np.asarray(
        [
            [
                int(float(row["bbox_x0"])),
                int(float(row["bbox_y0"])),
                int(float(row["bbox_x1"])),
                int(float(row["bbox_y1"])),
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    purities = np.asarray([float(row.get("gt_purity", row.get("gt_target_fraction", 0.0))) for row in rows], dtype=np.float32)
    valid_fraction = np.asarray([float(row.get("valid_fraction", 0.0)) for row in rows], dtype=np.float32)
    return segment_ids, areas, centers, boxes, labels, purities, valid_fraction


def compute_gt_counts(segments: np.ndarray, gt: np.ndarray, num_segments: int, chunk_rows: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if segments.shape != gt.shape:
        raise ValueError(f"shape mismatch: segments={segments.shape} gt={gt.shape}")
    counts = np.zeros((num_segments, NUM_CLASSES), dtype=np.int64)
    valid_pixels = np.zeros(num_segments, dtype=np.int64)
    for y0 in range(0, segments.shape[0], chunk_rows):
        y1 = min(segments.shape[0], y0 + chunk_rows)
        seg_block = np.asarray(segments[y0:y1])
        gt_block = np.asarray(gt[y0:y1])
        mask = (seg_block >= 0) & (gt_block >= 0) & (gt_block < NUM_CLASSES)
        if not np.any(mask):
            continue
        seg_ids = seg_block[mask].astype(np.int64, copy=False)
        labels = gt_block[mask].astype(np.int64, copy=False)
        flat = seg_ids * NUM_CLASSES + labels
        bincount = np.bincount(flat, minlength=num_segments * NUM_CLASSES)
        counts += bincount.reshape(num_segments, NUM_CLASSES)
        valid_pixels += np.bincount(seg_ids, minlength=num_segments)
    return counts, valid_pixels


def load_scale_data(image_id: str, scale: str, token_filename: str = "tokens_image_cell_reg_texture_cellstats.npy") -> ScaleData:
    scale_dir = MULTISCALE_ROOT / image_id / scale
    segment_ids, areas, centers, boxes, labels, purities, valid_fraction = load_superpixel_rows(scale_dir / "superpixels.csv")
    segments = np.load(scale_dir / "superpixels.npy", mmap_mode="r")
    tokens = np.asarray(np.load(scale_dir / token_filename, mmap_mode="r"), dtype=np.float32)
    if len(segment_ids) != tokens.shape[0]:
        raise ValueError(f"{image_id}/{scale}: superpixel rows != token rows")
    valid = (labels >= 0) & (labels < NUM_CLASSES) & (valid_fraction > 0)
    adjacency_path = scale_dir / "adjacency.npy"
    if adjacency_path.exists():
        edges = np.asarray(np.load(adjacency_path, mmap_mode="r"), dtype=np.int64)
        neighbors = [set() for _ in range(len(segment_ids))]
        for left, right in edges:
            if 0 <= int(left) < len(neighbors) and 0 <= int(right) < len(neighbors):
                neighbors[int(left)].add(int(right))
                neighbors[int(right)].add(int(left))
        adjacency = [np.asarray(sorted(items), dtype=np.int64) for items in neighbors]
    else:
        adjacency = segment_adjacency(np.asarray(segments), len(segment_ids))
        edge_items = []
        for left, items in enumerate(adjacency):
            for right in items:
                if left < int(right):
                    edge_items.append((left, int(right)))
        edges = np.asarray(edge_items, dtype=np.int64) if edge_items else np.empty((0, 2), dtype=np.int64)
    return ScaleData(
        image_id=image_id,
        scale=scale,
        segment_ids=segment_ids,
        tokens=tokens,
        tokens_norm=safe_l2_normalize(tokens, axis=1),
        areas=areas,
        centers=centers,
        boxes=boxes,
        valid=valid,
        gt_majority_label=labels,
        gt_purity=purities,
        valid_fraction=valid_fraction,
        adjacency=adjacency,
        edges=edges,
    )


def select_by_boxes(data: ScaleData, boxes: list[tuple[int, int, int, int]]) -> list[int]:
    selected: set[int] = set()
    for x0, y0, x1, y1 in boxes:
        center_hit = (
            (data.centers[:, 0] >= x0)
            & (data.centers[:, 0] < x1)
            & (data.centers[:, 1] >= y0)
            & (data.centers[:, 1] < y1)
            & data.valid
        )
        hit_ids = np.flatnonzero(center_hit)
        if hit_ids.size == 0:
            overlap = (
                (data.boxes[:, 0] < x1)
                & (data.boxes[:, 2] > x0)
                & (data.boxes[:, 1] < y1)
                & (data.boxes[:, 3] > y0)
                & data.valid
            )
            hit_ids = np.flatnonzero(overlap)
        selected.update(int(value) for value in hit_ids.tolist())
    return sorted(selected)


def prompt_segments_for_scale(prompt: dict[str, str], data: ScaleData) -> tuple[list[int], list[int]]:
    positive_boxes = parse_boxes(prompt.get("positive_boxes"))
    negative_boxes = parse_boxes(prompt.get("negative_boxes"))
    if data.scale == "medium":
        pos = [sid for sid in parse_ids(prompt.get("positive_segments")) if 0 <= sid < len(data.segment_ids)]
        neg = [sid for sid in parse_ids(prompt.get("negative_segments")) if 0 <= sid < len(data.segment_ids)]
        if pos:
            return pos, neg
    pos = select_by_boxes(data, positive_boxes)
    neg = select_by_boxes(data, negative_boxes)
    return pos, neg


def max_cosine_scores(tokens_norm: np.ndarray, prompt_ids: list[int]) -> np.ndarray:
    if not prompt_ids:
        return np.zeros(tokens_norm.shape[0], dtype=np.float32)
    ids = np.asarray(prompt_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < tokens_norm.shape[0])]
    if ids.size == 0:
        return np.zeros(tokens_norm.shape[0], dtype=np.float32)
    scores = tokens_norm @ tokens_norm[ids].T
    return np.max(scores, axis=1).astype(np.float32, copy=False)


def ranking_metrics(target: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if target.size == 0 or np.sum(target) == 0 or np.sum(~target) == 0:
        return 0.0, 0.5
    order = np.argsort(-scores, kind="stable")
    y = target[order].astype(np.float64)
    positives = float(np.sum(y))
    precision = np.cumsum(y) / (np.arange(len(y), dtype=np.float64) + 1.0)
    ap = float(np.sum(precision * y) / max(positives, 1.0))
    pos_scores = scores[target]
    neg_scores = scores[~target]
    sorted_neg = np.sort(neg_scores)
    less = np.searchsorted(sorted_neg, pos_scores, side="left")
    leq = np.searchsorted(sorted_neg, pos_scores, side="right")
    auc = float(np.mean((less + leq) / 2.0 / max(len(sorted_neg), 1)))
    return ap, auc


def precision_at_k(target: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    top = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    return float(np.mean(np.asarray(target, dtype=bool)[top])) if len(top) else 0.0


def best_area_metrics(target: np.ndarray, scores: np.ndarray, areas: np.ndarray) -> tuple[float, float]:
    best_dice = -1.0
    best_area = 0.0
    for fraction in FRACTIONS:
        pred = predict_top_area(scores, areas, fraction)
        dice = float(binary_segmentation_metrics(target, pred)["dice"])
        if dice > best_dice:
            best_dice = dice
            best_area = fraction
    return (best_dice if best_dice >= 0 else 0.0), best_area


def fast_gmm_mask(scores: np.ndarray, iterations: int = 12) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.zeros(len(values), dtype=bool)
    clean = values[finite]
    if clean.size < 4 or float(clean.min()) == float(clean.max()):
        output[finite] = clean >= percentile_threshold(clean, 90.0)
        return output
    centers = np.asarray([np.percentile(clean, 30.0), np.percentile(clean, 90.0)], dtype=np.float64)
    labels = np.zeros(clean.shape[0], dtype=np.int8)
    for _ in range(iterations):
        labels = np.argmin(np.abs(clean[:, None] - centers[None, :]), axis=1).astype(np.int8)
        new_centers = centers.copy()
        for cluster in (0, 1):
            if np.any(labels == cluster):
                new_centers[cluster] = float(np.mean(clean[labels == cluster]))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    positive_cluster = int(np.argmax(centers))
    output[finite] = labels == positive_cluster
    return output


def candidate_edges(edges: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edge_mask = candidate[edges[:, 0]] & candidate[edges[:, 1]]
    selected = edges[edge_mask]
    if selected.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    local = np.full(candidate.shape[0], -1, dtype=np.int64)
    local[np.flatnonzero(candidate)] = np.arange(int(np.sum(candidate)), dtype=np.int64)
    return local[selected]


def expand_boundary(boundary: np.ndarray, edges: np.ndarray, hops: int) -> np.ndarray:
    expanded = np.asarray(boundary, dtype=bool).copy()
    frontier = expanded.copy()
    for _ in range(hops):
        if not np.any(frontier) or edges.size == 0:
            break
        next_frontier = np.zeros_like(expanded)
        hit_left = frontier[edges[:, 0]]
        hit_right = frontier[edges[:, 1]]
        next_frontier[edges[hit_left, 1]] = True
        next_frontier[edges[hit_right, 0]] = True
        frontier = next_frontier & ~expanded
        expanded |= next_frontier
    return expanded


def graph_boundary_f1_edges(target: np.ndarray, pred: np.ndarray, edges: np.ndarray, hops: int) -> float:
    target = np.asarray(target, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    if edges.size == 0:
        return 1.0 if np.array_equal(target, pred) else 0.0
    target_boundary = np.zeros(len(target), dtype=bool)
    pred_boundary = np.zeros(len(pred), dtype=bool)
    target_diff = target[edges[:, 0]] != target[edges[:, 1]]
    pred_diff = pred[edges[:, 0]] != pred[edges[:, 1]]
    target_boundary[edges[target_diff, 0]] = True
    target_boundary[edges[target_diff, 1]] = True
    pred_boundary[edges[pred_diff, 0]] = True
    pred_boundary[edges[pred_diff, 1]] = True
    if not np.any(target_boundary) and not np.any(pred_boundary):
        return 1.0
    if not np.any(target_boundary) or not np.any(pred_boundary):
        return 0.0
    target_expanded = expand_boundary(target_boundary, edges, hops)
    pred_expanded = expand_boundary(pred_boundary, edges, hops)
    precision = float(np.sum(pred_boundary & target_expanded) / max(np.sum(pred_boundary), 1))
    recall = float(np.sum(target_boundary & pred_expanded) / max(np.sum(target_boundary), 1))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_area_metrics(target: np.ndarray, pred: np.ndarray, areas: np.ndarray) -> dict[str, float]:
    base = binary_segmentation_metrics(target, pred)
    target = np.asarray(target, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    areas = np.asarray(areas, dtype=np.float64)
    return {
        "dice": float(base["dice"]),
        "iou": float(base["iou"]),
        "precision": float(base["precision"]),
        "recall": float(base["recall"]),
        "pred_area": float(np.sum(areas[pred])),
        "gt_area": float(np.sum(areas[target])),
        "fp_area": float(np.sum(areas[pred & ~target])),
        "fn_area": float(np.sum(areas[~pred & target])),
    }


def evaluate_prompt_scale(
    prompt: dict[str, str],
    data: ScaleData,
    global_area_fraction: float,
    classwise_area_fraction: float,
    lambda_neg: float,
    force: bool,
) -> dict[str, object]:
    target_class = int(prompt["target_class"])
    pos_ids, neg_ids = prompt_segments_for_scale(prompt, data)
    out_path = SCORE_DIR / f"{prompt['query_id']}_{data.scale}.npz"
    hard = data.gt_majority_label.astype(np.int16) == target_class
    soft = np.where(hard, data.gt_purity * data.valid_fraction, 0.0).astype(np.float32)
    score_pos = max_cosine_scores(data.tokens_norm, pos_ids)
    score_neg = max_cosine_scores(data.tokens_norm, neg_ids)
    score_final = score_pos - float(lambda_neg) * score_neg if neg_ids else score_pos.copy()
    valid = data.valid
    candidate = valid.copy()
    status = "ok"
    if not pos_ids:
        status = "missing_positive_prompt"
    elif int(np.sum(candidate & hard)) == 0 or int(np.sum(candidate & ~hard)) == 0:
        status = "degenerate_target"

    ranks = np.empty(len(score_final), dtype=np.float32)
    order = np.argsort(score_final, kind="stable")
    ranks[order] = np.linspace(0.0, 1.0, len(score_final), dtype=np.float32)
    if force or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            segment_ids=data.segment_ids.astype(np.int64),
            score_pos=score_pos.astype(np.float32),
            score_neg=score_neg.astype(np.float32),
            score_final=score_final.astype(np.float32),
            rank_percentile=ranks,
            gt_soft_label_per_superpixel=soft,
            gt_hard_label_per_superpixel=hard.astype(np.uint8),
            area=data.areas.astype(np.float32),
            center_xy=data.centers.astype(np.float32),
            positive_prompt_segments=np.asarray(pos_ids, dtype=np.int64),
            negative_prompt_segments=np.asarray(neg_ids, dtype=np.int64),
        )

    row: dict[str, object] = {
        "query_id": prompt["query_id"],
        "image_id": prompt["image_id"],
        "target_class": target_class,
        "scale": data.scale,
        "prompt_quality": prompt.get("prompt_quality", ""),
        "prompt_mode": prompt.get("prompt_mode", ""),
        "positive_prompt_segments": len(pos_ids),
        "negative_prompt_segments": len(neg_ids),
        "candidate_segments": int(np.sum(candidate)),
        "status": status,
    }
    if status != "ok":
        for key in METRIC_FIELDS:
            row.setdefault(key, "")
        return row

    scores = score_final[candidate]
    target = hard[candidate]
    areas = data.areas[candidate]
    local_edges = candidate_edges(data.edges, candidate)

    ap, auroc = ranking_metrics(target, scores)
    p80 = scores >= percentile_threshold(scores, 80.0)
    p90 = scores >= percentile_threshold(scores, 90.0)
    otsu = scores >= otsu_score_threshold(scores)
    gmm = fast_gmm_mask(scores)
    global_top = predict_top_area(scores, areas, global_area_fraction)
    classwise_top = predict_top_area(scores, areas, classwise_area_fraction)
    best_dice, best_area = best_area_metrics(target, scores, areas)
    primary = binary_area_metrics(target, classwise_top, areas)

    row.update(
        {
            "mAP": ap,
            "AUROC": auroc,
            "P@top5": precision_at_k(target, scores, 5),
            "P@top10": precision_at_k(target, scores, 10),
            "Dice_p90": float(binary_segmentation_metrics(target, p90)["dice"]),
            "Dice_p80": float(binary_segmentation_metrics(target, p80)["dice"]),
            "Dice_otsu": float(binary_segmentation_metrics(target, otsu)["dice"]),
            "Dice_gmm": float(binary_segmentation_metrics(target, gmm)["dice"]),
            "Dice_global_toparea": float(binary_segmentation_metrics(target, global_top)["dice"]),
            "Dice_classwise_toparea": primary["dice"],
            "BestDice": best_dice,
            "BestArea": best_area,
            "mIoU": primary["iou"],
            "BF1@5": graph_boundary_f1_edges(target, classwise_top, local_edges, 1),
            "BF1@10": graph_boundary_f1_edges(target, classwise_top, local_edges, 2),
            "PredArea": primary["pred_area"],
            "GTArea": primary["gt_area"],
            "Precision": primary["precision"],
            "Recall": primary["recall"],
            "FP_area": primary["fp_area"],
            "FN_area": primary["fn_area"],
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_p90": float(np.percentile(scores, 90.0)),
            "gt_positive_segments": int(np.sum(target)),
            "gt_negative_segments": int(np.sum(~target)),
        }
    )
    return row


def class_area_floor(class_id: int, scale: str) -> float:
    return CLASS_SCALE_AREA_FLOORS.get((class_id, scale), DEFAULT_AREA_FLOOR)


def class_area_priors(manifest_rows: list[dict[str, str]]) -> dict[tuple[str, int, str], tuple[float, float]]:
    priors: dict[tuple[str, int, str], tuple[float, float]] = {}
    by_scale_class: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for row in manifest_rows:
        image_id = row["image_id"]
        for scale in SCALES:
            records = read_csv(MULTISCALE_ROOT / image_id / scale / "superpixels.csv")
            valid_area = 0.0
            target_area_by_class = np.zeros(NUM_CLASSES, dtype=np.float64)
            for record in records:
                label = int(float(record.get("gt_majority_label", 255)))
                area = float(record.get("area", 0.0))
                valid_fraction = float(record.get("valid_fraction", 0.0))
                if valid_fraction <= 0:
                    continue
                valid_area += area
                if 0 <= label < NUM_CLASSES:
                    target_area_by_class[label] += area
            for class_id in range(NUM_CLASSES):
                by_scale_class[(scale, class_id)].append((image_id, float(target_area_by_class[class_id]) / max(valid_area, 1.0)))
    for (scale, class_id), values in by_scale_class.items():
        floor = class_area_floor(class_id, scale)
        all_fracs = [value for _, value in values]
        global_median = float(np.clip(np.median(all_fracs), floor, 0.60)) if all_fracs else 0.18
        for image_id, _ in values:
            loio = [value for other_image, value in values if other_image != image_id]
            classwise = float(np.clip(np.median(loio), floor, 0.60)) if loio else global_median
            priors[(image_id, class_id, scale)] = (global_median, classwise)
    return priors


def evaluate_image(
    image_id: str,
    manifest_row: dict[str, str],
    prompts: list[dict[str, str]],
    priors: dict[tuple[str, int, str], tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    scale_data = {scale: load_scale_data(image_id, scale, args.token_filename) for scale in SCALES}
    rows: list[dict[str, object]] = []
    for prompt in prompts:
        target_class = int(prompt["target_class"])
        for scale, data in scale_data.items():
            global_prior, classwise_prior = priors.get((image_id, target_class, scale), (0.18, 0.18))
            rows.append(evaluate_prompt_scale(prompt, data, global_prior, classwise_prior, args.lambda_neg, args.force))
    report = {
        "image_id": image_id,
        "prompts": len(prompts),
        "query_scale_rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in rows)),
    }
    return image_id, rows, report


def summarize(
    rows: list[dict[str, object]],
    image_reports: list[dict[str, object]],
    prompt_count: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    errors: list[str] = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    expected = prompt_count * len(SCALES)
    score_files = len(list(SCORE_DIR.glob("*.npz"))) if SCORE_DIR.exists() else 0
    if len(rows) != expected:
        errors.append(f"metric row count {len(rows)} != expected {expected}")
    if score_files < expected:
        errors.append(f"score npz count {score_files} < expected {expected}")
    if not ok_rows:
        errors.append("no valid query-scale rows")
    if not METRICS_PATH.exists():
        errors.append(f"missing metrics CSV: {METRICS_PATH}")
    by_scale = Counter(str(row["scale"]) for row in ok_rows)
    return {
        "passed": not errors,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_tasks": str(args.prompts),
        "query_scale_score_dir": str(SCORE_DIR),
        "metrics_csv": str(METRICS_PATH),
        "total_metric_rows": len(rows),
        "ok_metric_rows": len(ok_rows),
        "expected_query_scale_rows": expected,
        "score_npz_files": score_files,
        "scale_counts": dict(by_scale),
        "image_reports": image_reports,
        "errors": errors[:50],
    }


def write_report(validation: dict[str, object], rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Phase C Multiscale Baseline Retrieval",
        "",
        "## Summary",
        "",
        "Phase C computes prompt-conditioned cosine retrieval scores for each Phase B query on small, medium, and large superpixels.",
        "GT hard/soft labels use Phase A standardized `superpixels.csv` fields for speed and consistency with Phase B prompt sampling.",
        "",
        "## Outputs",
        "",
        f"- query-scale scores: {SCORE_DIR}",
        f"- metrics CSV: {METRICS_PATH}",
        f"- validation JSON: {VALIDATION_PATH}",
        "",
        "## Validation",
        "",
        f"- passed: {validation['passed']}",
        f"- total_metric_rows: {validation['total_metric_rows']}",
        f"- ok_metric_rows: {validation['ok_metric_rows']}",
        f"- score_npz_files: {validation['score_npz_files']}",
        f"- scale_counts: {validation['scale_counts']}",
    ]
    if ok_rows:
        lines.extend(
            [
                "",
                "## Mean Metrics",
                "",
                "| scale | rows | mAP | AUROC | Dice_classwise_toparea | BestDice |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for scale in SCALES:
            subset = [row for row in ok_rows if row["scale"] == scale]
            if not subset:
                continue
            lines.append(
                f"| {scale} | {len(subset)} | "
                f"{np.mean([float(row['mAP']) for row in subset]):.4f} | "
                f"{np.mean([float(row['AUROC']) for row in subset]):.4f} | "
                f"{np.mean([float(row['Dice_classwise_toparea']) for row in subset]):.4f} | "
                f"{np.mean([float(row['BestDice']) for row in subset]):.4f} |"
            )
    if validation.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in validation["errors"])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--prompts", type=Path, default=PROMPT_PATH)
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--token_filename",
        default="tokens_image_cell_reg_texture_cellstats.npy",
        help="每个 scale 目录内使用的 token 文件；默认使用 texture/cellstats enhanced token。",
    )
    parser.add_argument("--max_queries", type=int, default=0, help="Debug limit; 0 means all queries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    manifest_by_image = {row["image_id"]: row for row in manifest_rows}
    prompts = read_csv(args.prompts)
    if args.max_queries > 0:
        prompts = prompts[: args.max_queries]
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in prompts:
        by_image[prompt["image_id"]].append(prompt)

    print("phase_c computing class area priors", flush=True)
    priors = class_area_priors(manifest_rows)
    all_rows: list[dict[str, object]] = []
    image_reports: list[dict[str, object]] = []
    jobs = [(image_id, manifest_by_image[image_id], by_image[image_id]) for image_id in sorted(by_image)]
    if args.workers <= 1:
        for index, (image_id, manifest_row, image_prompts) in enumerate(jobs, start=1):
            _, rows, report = evaluate_image(image_id, manifest_row, image_prompts, priors, args)
            all_rows.extend(rows)
            image_reports.append(report)
            print(f"phase_c {index}/{len(jobs)} image_id={image_id} rows={len(rows)} ok={report['ok_rows']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_image, image_id, manifest_row, image_prompts, priors, args): image_id
                for image_id, manifest_row, image_prompts in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                image_id, rows, report = future.result()
                all_rows.extend(rows)
                image_reports.append(report)
                print(f"phase_c {index}/{len(jobs)} image_id={image_id} rows={len(rows)} ok={report['ok_rows']}", flush=True)

    all_rows.sort(key=lambda row: (str(row["image_id"]), str(row["query_id"]), str(row["scale"])))
    write_csv(METRICS_PATH, all_rows, METRIC_FIELDS)
    validation = summarize(all_rows, sorted(image_reports, key=lambda row: str(row["image_id"])), len(prompts), args)
    write_json(VALIDATION_PATH, validation)
    write_report(validation, all_rows)
    if not validation["passed"]:
        raise SystemExit(f"Phase C validation failed: {validation['errors'][:5]}")
    print(json.dumps({"passed": True, "rows": len(all_rows), "validation": str(VALIDATION_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
