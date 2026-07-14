# 作用：在相同 v3 hard-negative query 上比较无负提示、随机异类负提示和受限难负提示，不训练模型。

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import binary_segmentation_metrics, predict_top_area
from benchmarks.v3.phase_c.run_multiscale_baseline import (
    METRICS_PATH,
    MULTISCALE_ROOT,
    NUM_CLASSES,
    SCALES,
    ScaleData,
    load_scale_data,
    max_cosine_scores,
    parse_ids,
    prompt_segments_for_scale,
    ranking_metrics,
    read_csv,
)


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
PROMPT_PATH = PRET_ROOT / "prompt_tasks" / "all_prompt_tasks.csv"
OUT_METRICS = PRET_ROOT / "evaluations" / "negative_prompt_controls_metrics.csv"
OUT_SUMMARY = PRET_ROOT / "evaluations" / "negative_prompt_controls_summary.csv"
REPORT_PATH = Path(__file__).resolve().parent / "negative_prompt_controls_report.md"

METRIC_FIELDS = [
    "strategy", "query_id", "image_id", "target_class", "scale", "negative_count",
    "negative_similarity_mean_medium", "negative_similarity_max_medium", "mAP", "AUROC",
    "Dice_classwise_toparea", "mIoU", "Precision", "Recall", "score_mean", "score_std",
    "score_p90", "area_fraction", "status",
]
SUMMARY_FIELDS = [
    "strategy", "scale", "n", "mAP", "AUROC", "Dice_classwise_toparea", "mIoU",
    "Precision", "Recall", "score_std", "negative_similarity_mean_medium",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])
    os.replace(tmp, path)


def stable_rng(*values: object) -> np.random.Generator:
    text = "|".join(map(str, values))
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def loio_bestarea_priors(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], float]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(int(row["target_class"]), row["scale"])].append(row)
    priors: dict[tuple[str, int, str], float] = {}
    for (class_id, scale), group in groups.items():
        fallback = float(np.median([float(row["BestArea"]) for row in group]))
        for row in group:
            train = [float(other["BestArea"]) for other in group if other["image_id"] != row["image_id"]]
            priors[(row["image_id"], class_id, scale)] = float(np.clip(np.median(train) if train else fallback, 0.01, 0.60))
    return priors


def medium_negative_candidates(data: ScaleData, target_class: int, positive_ids: list[int]) -> np.ndarray:
    ids = np.arange(len(data.segment_ids), dtype=np.int64)
    return ids[
        data.valid
        & (data.gt_majority_label != target_class)
        & (data.valid_fraction >= 0.5)
        & ~np.isin(ids, np.asarray(positive_ids, dtype=np.int64))
    ]


def choose_negatives(
    strategy: str,
    prompt: dict[str, str],
    medium: ScaleData,
    low_quantile: float,
    high_quantile: float,
) -> tuple[list[int], np.ndarray]:
    positives = parse_ids(prompt["positive_segments"])
    count = max(1, len(parse_ids(prompt["negative_segments"])))
    if strategy == "positive_only":
        return [], np.empty(0, dtype=np.float32)
    if strategy == "extreme_hard":
        selected = parse_ids(prompt["negative_segments"])
        sims = max_cosine_scores(medium.tokens_norm, positives)[np.asarray(selected, dtype=np.int64)]
        return selected, sims

    candidates = medium_negative_candidates(medium, int(prompt["target_class"]), positives)
    scores = max_cosine_scores(medium.tokens_norm, positives)
    if strategy == "random_negative":
        pool = candidates
    elif strategy == "bounded_hard_negative":
        lo, hi = np.quantile(scores[candidates], [low_quantile, high_quantile])
        pool = candidates[(scores[candidates] >= lo) & (scores[candidates] <= hi)]
        if len(pool) < count:
            pool = candidates
    else:
        raise ValueError(strategy)
    rng = stable_rng(prompt["query_id"], strategy)
    selected = rng.choice(pool, size=min(count, len(pool)), replace=False).astype(np.int64)
    return selected.tolist(), scores[selected]


def with_negative_boxes(prompt: dict[str, str], medium: ScaleData, negative_ids: list[int]) -> dict[str, str]:
    copied = dict(prompt)
    copied["negative_segments"] = ";".join(str(value) for value in negative_ids)
    copied["negative_boxes"] = ";".join(
        f"{int(medium.boxes[idx, 0])}:{int(medium.boxes[idx, 1])}:{int(medium.boxes[idx, 2])}:{int(medium.boxes[idx, 3])}"
        for idx in negative_ids
    )
    return copied


def evaluate(
    strategy: str,
    prompt: dict[str, str],
    data: ScaleData,
    medium: ScaleData,
    negative_ids: list[int],
    medium_sims: np.ndarray,
    area_fraction: float,
    lambda_neg: float,
) -> dict[str, object]:
    control = with_negative_boxes(prompt, medium, negative_ids)
    pos_ids, neg_ids = prompt_segments_for_scale(control, data)
    target_class = int(prompt["target_class"])
    hard = data.gt_majority_label == target_class
    candidate = data.valid
    if not pos_ids or not np.any(candidate & hard) or not np.any(candidate & ~hard):
        return {"strategy": strategy, "query_id": prompt["query_id"], "image_id": prompt["image_id"], "target_class": target_class, "scale": data.scale, "status": "invalid_query"}
    score_pos = max_cosine_scores(data.tokens_norm, pos_ids)
    score_neg = max_cosine_scores(data.tokens_norm, neg_ids)
    score = score_pos if not neg_ids else score_pos - lambda_neg * score_neg
    scores, labels, areas = score[candidate], hard[candidate], data.areas[candidate]
    pred = predict_top_area(scores, areas, area_fraction)
    binary = binary_segmentation_metrics(labels, pred)
    ap, auroc = ranking_metrics(labels, scores)
    return {
        "strategy": strategy,
        "query_id": prompt["query_id"],
        "image_id": prompt["image_id"],
        "target_class": target_class,
        "scale": data.scale,
        "negative_count": len(neg_ids),
        "negative_similarity_mean_medium": float(np.mean(medium_sims)) if len(medium_sims) else "",
        "negative_similarity_max_medium": float(np.max(medium_sims)) if len(medium_sims) else "",
        "mAP": ap,
        "AUROC": auroc,
        "Dice_classwise_toparea": float(binary["dice"]),
        "mIoU": float(binary["iou"]),
        "Precision": float(binary["precision"]),
        "Recall": float(binary["recall"]),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_p90": float(np.percentile(scores, 90.0)),
        "area_fraction": area_fraction,
        "status": "ok",
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(str(row["strategy"]), str(row["scale"]))].append(row)
    output = []
    for (strategy, scale), group in sorted(groups.items()):
        mean = lambda field: float(np.mean([float(row[field]) for row in group if row.get(field) != ""]))
        output.append({
            "strategy": strategy, "scale": scale, "n": len(group),
            "mAP": mean("mAP"), "AUROC": mean("AUROC"),
            "Dice_classwise_toparea": mean("Dice_classwise_toparea"), "mIoU": mean("mIoU"),
            "Precision": mean("Precision"), "Recall": mean("Recall"), "score_std": mean("score_std"),
            "negative_similarity_mean_medium": mean("negative_similarity_mean_medium") if strategy != "positive_only" else "",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--low_quantile", type=float, default=0.85)
    parser.add_argument("--high_quantile", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0, help="仅用于 smoke test；0 表示全量")
    args = parser.parse_args()
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")

    priors = loio_bestarea_priors(read_csv(METRICS_PATH))
    prompts = [row for row in read_csv(PROMPT_PATH) if row.get("prompt_quality") == "hard_negative"]
    if args.limit:
        prompts = prompts[:args.limit]
    cache: dict[str, dict[str, ScaleData]] = {}
    rows: list[dict[str, object]] = []
    strategies = ("extreme_hard", "positive_only", "random_negative", "bounded_hard_negative")
    for index, prompt in enumerate(prompts, start=1):
        image_id = prompt["image_id"]
        if image_id not in cache:
            cache[image_id] = {scale: load_scale_data(image_id, scale) for scale in SCALES}
        scale_data = cache[image_id]
        medium = scale_data["medium"]
        for strategy in strategies:
            negative_ids, medium_sims = choose_negatives(strategy, prompt, medium, args.low_quantile, args.high_quantile)
            for scale, data in scale_data.items():
                prior = priors[(image_id, int(prompt["target_class"]), scale)]
                rows.append(evaluate(strategy, prompt, data, medium, negative_ids, medium_sims, prior, args.lambda_neg))
        if index % 100 == 0 or index == len(prompts):
            print(f"negative_controls {index}/{len(prompts)}", flush=True)
    summary = summarize(rows)
    write_csv(OUT_METRICS, rows, METRIC_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    lines = ["# Phase C Negative Prompt Controls", "", "- paired hard-negative queries: " + str(len(prompts)), "- strategies: extreme_hard, positive_only, random_negative, bounded_hard_negative", f"- bounded hard quantiles: {args.low_quantile:.2f}-{args.high_quantile:.2f}", "", "| strategy | scale | n | mAP | AUROC | Dice | mIoU | score std |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['strategy']} | {row['scale']} | {row['n']} | {row['mAP']:.4f} | {row['AUROC']:.4f} | {row['Dice_classwise_toparea']:.4f} | {row['mIoU']:.4f} | {row['score_std']:.4f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_METRICS}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
