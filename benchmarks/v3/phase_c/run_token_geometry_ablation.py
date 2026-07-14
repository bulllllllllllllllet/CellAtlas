# 作用：严格按 12/4/4 WSI 切分评估 token 归一化、block fusion 和 prompt aggregation 对小尺度 retrieval 的影响。

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import safe_l2_normalize
from benchmarks.v3.phase_c.run_multiscale_baseline import METRICS_PATH, MULTISCALE_ROOT, SCORE_DIR, read_csv, ranking_metrics
from benchmarks.v3.phase_e.train_mlp_mask_decoder import MANIFEST, load_image

PRET_ROOT = MULTISCALE_ROOT.parent
OUT = PRET_ROOT / "evaluations"
PREDICTIONS_OUT = OUT / "token_geometry_ablation_oof.csv"
FRACTIONS_OUT = OUT / "token_geometry_ablation_fractions.csv"
SUMMARY_OUT = OUT / "token_geometry_ablation_summary.csv"
VALIDATION_OUT = OUT / "token_geometry_ablation_validation.json"
PROGRESS_OUT = OUT / "token_geometry_ablation_progress.json"
CHECKPOINT_DIR = OUT / "token_geometry_ablation_checkpoints"
REPORT_OUT = Path(__file__).resolve().parent / "token_geometry_ablation_report.md"
SCALE = "small"
FRACTIONS = np.linspace(0.01, 0.60, 60, dtype=np.float64)
FUSIONS = {
    "fusion_current": (1.0, 0.5, 0.75, 0.5),
    "fusion_imageheavy": (1.0, 0.25, 0.75, 0.5),
    "fusion_cellheavy": (1.0, 1.0, 0.75, 0.5),
    "fusion_textureheavy": (1.0, 0.5, 1.25, 0.5),
    "fusion_nocell": (1.0, 0.0, 0.75, 0.5),
    "fusion_noimage": (0.0, 1.0, 0.75, 0.5),
}


@dataclass
class Curve:
    score_mean: float
    score_std: float
    fraction_gt_098: float
    cumulative_target: np.ndarray
    cumulative_pred: np.ndarray
    cumulative_area: np.ndarray
    gt_area: float
    ordered: np.ndarray
    ordered_scores: np.ndarray


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_paths(test_fold: int) -> tuple[Path, Path, Path]:
    stem = f"fold_{test_fold}"
    return (
        CHECKPOINT_DIR / f"{stem}_oof.csv",
        CHECKPOINT_DIR / f"{stem}_fractions.csv",
        CHECKPOINT_DIR / f"{stem}_meta.json",
    )


def source_dir(image_id: str) -> Path:
    payload = json.loads((MULTISCALE_ROOT / image_id / SCALE / "validation.json").read_text(encoding="utf-8"))
    return Path(payload["source_dir"])


def ranking_labels(image_id: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(MULTISCALE_ROOT / image_id / SCALE / "superpixels.csv")
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    fraction = np.asarray([float(row.get("valid_fraction", 0.0)) for row in rows], dtype=np.float32)
    valid = (labels >= 0) & (labels < 12) & (fraction > 0)
    return labels, valid


def valid_mask(image_id: str) -> np.ndarray:
    return ranking_labels(image_id)[1]


def normalize(values: np.ndarray) -> np.ndarray:
    return safe_l2_normalize(np.asarray(values, dtype=np.float32), axis=1)


class FoldTransform:
    def __init__(self, tokens: np.ndarray) -> None:
        self.mean = tokens.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(tokens.std(axis=0, keepdims=True), 1e-5).astype(np.float32)
        centered = tokens - self.mean
        covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        self.values = values[order].astype(np.float32)
        self.vectors = vectors[:, order].astype(np.float32)
        ratio = np.cumsum(self.values) / max(float(self.values.sum()), 1e-12)
        self.white_dim = min(128, max(1, int(np.searchsorted(ratio, 0.99, side="left")) + 1))

    def apply(self, tokens: np.ndarray, name: str) -> np.ndarray:
        raw = normalize(tokens)
        centered = raw - self.mean
        if name == "enhanced_raw":
            return raw
        if name == "enhanced_centered":
            return normalize(centered)
        if name == "enhanced_zscore":
            return normalize(centered / self.std)
        if name in {"enhanced_remove_pc1", "enhanced_remove_pc3"}:
            count = 1 if name.endswith("pc1") else 3
            basis = self.vectors[:, :count]
            return normalize(centered - (centered @ basis) @ basis.T)
        if name == "enhanced_whiten":
            basis = self.vectors[:, :self.white_dim]
            floor = max(float(self.values[0]) * 1e-3, 1e-8)
            projected = centered @ basis / np.sqrt(np.maximum(self.values[:self.white_dim], floor))
            return normalize(projected)
        raise ValueError(name)


def fit_transform(image_ids: list[str]) -> FoldTransform:
    samples = []
    for image_id in image_ids:
        values = np.asarray(np.load(source_dir(image_id) / "tokens_image_cell_reg_texture_cellstats.npy", mmap_mode="r"), dtype=np.float32)
        samples.append(normalize(values)[valid_mask(image_id)])
    return FoldTransform(np.concatenate(samples, axis=0))


def blocks_for_image(image_id: str, transform: FoldTransform) -> dict[str, np.ndarray]:
    directory = source_dir(image_id)
    enhanced = np.asarray(np.load(directory / "tokens_image_cell_reg_texture_cellstats.npy", mmap_mode="r"), dtype=np.float32)
    image = normalize(np.load(directory / "tokens_image_only.npy", mmap_mode="r"))
    cell = normalize(np.load(directory / "tokens_cell_reg.npy", mmap_mode="r"))
    texture = normalize(enhanced[:, 274:304])
    cellstats = normalize(enhanced[:, 304:309])
    output = {name: transform.apply(enhanced, name) for name in (
        "enhanced_raw", "enhanced_centered", "enhanced_zscore", "enhanced_remove_pc1", "enhanced_remove_pc3", "enhanced_whiten",
    )}
    output.update({"image": image, "cell": cell, "texture": texture, "cellstats": cellstats})
    for name, weights in FUSIONS.items():
        parts = [block * np.float32(weight) for block, weight in zip((image, cell, texture, cellstats), weights, strict=True) if weight > 0]
        output[name] = normalize(np.concatenate(parts, axis=1))
    return output


def candidate_specs() -> list[tuple[str, str, str]]:
    base = [
        "enhanced_raw", "enhanced_centered", "enhanced_zscore", "enhanced_remove_pc1", "enhanced_remove_pc3", "enhanced_whiten",
        "image", "cell", "texture", "cellstats", *FUSIONS.keys(),
    ]
    specs = [(name, name, "max") for name in base]
    for token_name in ("enhanced_raw", "fusion_current"):
        for aggregation in ("mean", "median", "top2mean", "logsumexp"):
            specs.append((f"{token_name}_{aggregation}", token_name, aggregation))
    return specs


def aggregate(similarity: torch.Tensor, method: str) -> torch.Tensor:
    if method == "max":
        return similarity.max(dim=1).values
    if method == "mean":
        return similarity.mean(dim=1)
    if method == "median":
        return similarity.median(dim=1).values
    if method == "top2mean":
        return similarity.topk(k=min(2, similarity.shape[1]), dim=1).values.mean(dim=1)
    if method == "logsumexp":
        temperature = 0.05
        return temperature * (torch.logsumexp(similarity / temperature, dim=1) - np.log(similarity.shape[1]))
    raise ValueError(method)


def curve(data, class_id: int, scores: np.ndarray) -> Curve:
    valid = data.valid_pixels > 0
    ids = np.flatnonzero(valid)
    order = np.argsort(-scores[ids], kind="stable")
    ordered = ids[order]
    local = scores[valid]
    return Curve(
        score_mean=float(local.mean()), score_std=float(local.std()), fraction_gt_098=float(np.mean(local > 0.98)),
        cumulative_target=np.cumsum(data.counts[ordered, class_id], dtype=np.float64),
        cumulative_pred=np.cumsum(data.valid_pixels[ordered], dtype=np.float64),
        cumulative_area=np.cumsum(data.areas[ordered], dtype=np.float64),
        gt_area=float(data.counts[:, class_id].sum()),
        ordered=ordered,
        ordered_scores=scores[ordered].astype(np.float64, copy=False),
    )


def metric(curve_value: Curve, fraction: float) -> dict[str, float]:
    limit = curve_value.cumulative_area[-1] * fraction
    count = min(int(np.searchsorted(curve_value.cumulative_area, limit, side="left")) + 1, len(curve_value.cumulative_area))
    tp = float(curve_value.cumulative_target[count - 1])
    predicted = float(curve_value.cumulative_pred[count - 1])
    fp, fn = predicted - tp, curve_value.gt_area - tp
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "PixelDice": 2 * tp / denominator if denominator else 1.0,
        "Pixel_mIoU": tp / union if union else 1.0,
        "PixelPrecision": tp / (tp + fp) if tp + fp else 1.0,
        "PixelRecall": tp / (tp + fn) if tp + fn else 1.0,
    }


def dice_grid(curve_value: Curve) -> np.ndarray:
    limits = curve_value.cumulative_area[-1] * FRACTIONS
    indices = np.minimum(np.searchsorted(curve_value.cumulative_area, limits, side="left"), len(curve_value.cumulative_area) - 1)
    tp = curve_value.cumulative_target[indices]
    predicted = curve_value.cumulative_pred[indices]
    denominator = predicted + curve_value.gt_area
    return np.divide(2.0 * tp, denominator, out=np.ones_like(tp), where=denominator > 0)


def mean_dice_grid(grids: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(grid, dtype=np.float64).reshape(-1) for grid in grids], axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != len(FRACTIONS):
        raise RuntimeError(f"invalid calibration grid shape: {stacked.shape}, expected (*, {len(FRACTIONS)})")
    return stacked.mean(axis=0)


def ranking_metrics_from_order(target: np.ndarray, ordered_scores: np.ndarray) -> tuple[float, float]:
    """Exact AP/AUROC using the score order already needed for top-area pixel metrics."""
    target = np.asarray(target, dtype=bool)
    positives = int(target.sum())
    negatives = len(target) - positives
    if not len(target) or not positives or not negatives:
        return 0.0, 0.5
    y = target.astype(np.float64)
    ap = float(np.sum((np.cumsum(y) / (np.arange(len(y), dtype=np.float64) + 1.0)) * y) / positives)
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores) != 0) + 1]
    ends = np.r_[starts[1:], len(target)]
    group_positive = np.add.reduceat(target.astype(np.int64), starts)
    group_negative = (ends - starts) - group_positive
    seen_negative = np.r_[0, np.cumsum(group_negative[:-1])]
    lower_negative = negatives - seen_negative - group_negative
    wins = float(np.sum(group_positive * (lower_negative + 0.5 * group_negative)))
    return ap, float(wins / (positives * negatives))


def upload_tokens(tokens: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Keep each image's candidate representations resident for all of its prompts."""
    return {name: torch.from_numpy(values).to(device) for name, values in tokens.items()}


def score_candidates(tokens: dict[str, torch.Tensor], positive: np.ndarray, negative: np.ndarray, specs: list[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    output = {}
    size = next(iter(tokens.values())).shape[0]
    positive = positive[(positive >= 0) & (positive < size)]
    negative = negative[(negative >= 0) & (negative < size)]
    if not len(positive):
        raise RuntimeError("prompt has no valid positive superpixels")
    device = next(iter(tokens.values())).device
    positive_ids = torch.from_numpy(positive).to(device)
    negative_ids = torch.from_numpy(negative).to(device) if len(negative) else None
    for name, token_name, aggregation in specs:
        values = tokens[token_name]
        pos = aggregate(values @ values[positive_ids].T, aggregation)
        if negative_ids is not None:
            neg = aggregate(values @ values[negative_ids].T, aggregation)
            pos = pos - 0.5 * neg
        output[name] = pos.cpu().numpy().astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=-1, help="Debug single outer test fold; -1 runs all five.")
    parser.add_argument("--calibration_fold_offset", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this ablation")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    manifest_rows = read_csv(MANIFEST)
    gt_paths = {row["image_id"]: row["gt_mask_path"] for row in manifest_rows}
    image_folds = {row["image_id"]: int(row["fold"]) for row in manifest_rows}
    rows = [row for row in read_csv(METRICS_PATH) if row["scale"] == SCALE and row["status"] == "ok"]
    if len(rows) != 3711:
        raise RuntimeError(f"expected 3711 small rows, got {len(rows)}")
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    folds = sorted(set(image_folds.values()))
    selected_folds = folds if args.fold < 0 else [args.fold]
    specs = candidate_specs()
    oof_rows: list[dict[str, object]] = []
    fraction_rows: list[dict[str, object]] = []
    split_audit = []
    total_image_passes = sum(
        2 * sum(1 for fold in image_folds.values() if fold == test_fold)
        for test_fold in selected_folds
    )
    completed_image_passes = 0
    completed_queries = 0
    pending_folds = []
    for test_fold in selected_folds:
        expected_rows = sum(len(by_image[image_id]) for image_id, fold in image_folds.items() if fold == test_fold) * len(specs)
        oof_path, fractions_path, meta_path = checkpoint_paths(test_fold)
        if args.fold < 0 and oof_path.exists() and fractions_path.exists() and meta_path.exists():
            cached_oof = read_csv(oof_path)
            if len(cached_oof) == expected_rows:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                oof_rows.extend(cached_oof)
                fraction_rows.extend(read_csv(fractions_path))
                split_audit.append(meta["split_audit"])
                completed_image_passes += meta["image_passes"]
                completed_queries += meta["query_passes"]
                continue
        pending_folds.append(test_fold)
    write_json(PROGRESS_OUT, {"status": "running", "completed_image_passes": completed_image_passes, "total_image_passes": total_image_passes, "completed_queries": completed_queries, "total_queries": len(rows) * (2 if args.fold < 0 else 1), "resumed_folds": sorted(set(selected_folds) - set(pending_folds))})

    for test_fold in pending_folds:
        calibration_fold = folds[(folds.index(test_fold) + args.calibration_fold_offset) % len(folds)]
        train_folds = set(folds) - {test_fold, calibration_fold}
        train_images = [image_id for image_id, fold in image_folds.items() if fold in train_folds]
        transform = fit_transform(train_images)
        fold_audit = {"test_fold": test_fold, "calibration_fold": calibration_fold, "train_images": sorted(train_images), "calibration_images": sorted(image_id for image_id, fold in image_folds.items() if fold == calibration_fold), "test_images": sorted(image_id for image_id, fold in image_folds.items() if fold == test_fold)}
        split_audit.append(fold_audit)
        calibration_grids: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        calibration_global_grids: dict[str, list[np.ndarray]] = defaultdict(list)
        image_data_cache = {}
        fold_oof_start = len(oof_rows)
        fold_fractions_start = len(fraction_rows)

        for phase, fold in (("calibration", calibration_fold), ("test", test_fold)):
            for image_id in sorted(image_id for image_id, value in image_folds.items() if value == fold):
                tokens = upload_tokens(blocks_for_image(image_id, transform), device)
                data = load_image(image_id, gt_paths[image_id], image_data_cache)
                majority_labels, majority_valid = ranking_labels(image_id)
                if len(majority_labels) != len(data.valid_pixels):
                    raise RuntimeError(f"superpixel row mismatch for {image_id}")
                shared_ranking_order = np.array_equal(majority_valid, data.valid_pixels > 0)
                for row in by_image[image_id]:
                    with np.load(SCORE_DIR / f"{row['query_id']}_{SCALE}.npz") as prompt:
                        scores = score_candidates(tokens, prompt["positive_prompt_segments"].astype(np.int64), prompt["negative_prompt_segments"].astype(np.int64), specs)
                    class_id = int(row["target_class"])
                    completed_queries += 1
                    if phase == "calibration":
                        for name, values in scores.items():
                            grid = dice_grid(curve(data, class_id, values))
                            calibration_grids[(name, class_id)].append(grid)
                            calibration_global_grids[name].append(grid)
                    else:
                        # Fractions are available after calibration; defer test rows to a compact in-memory record.
                        for name, values in scores.items():
                            key = (name, class_id)
                            grids = calibration_grids[key]
                            if not grids:
                                grids = calibration_global_grids[name]
                            if not grids:
                                raise RuntimeError(f"no calibration grids for candidate={name}")
                            mean_grid = mean_dice_grid(grids)
                            best_index = int(np.argmax(mean_grid))
                            fraction = float(FRACTIONS[best_index])
                            current = curve(data, class_id, values)
                            calibrated = metric(current, fraction)
                            best = float(np.max(dice_grid(current)))
                            if shared_ranking_order:
                                ranking_map, ranking_auc = ranking_metrics_from_order(
                                    majority_labels[current.ordered] == class_id,
                                    current.ordered_scores,
                                )
                            else:
                                ranking_target = majority_labels[majority_valid] == class_id
                                ranking_map, ranking_auc = ranking_metrics(ranking_target, values[majority_valid])
                            oof_rows.append({"candidate": name, "fold": test_fold, "calibration_fold": calibration_fold, "query_id": row["query_id"], "image_id": image_id, "target_class": class_id, "prompt_quality": row["prompt_quality"], "prompt_mode": row["prompt_mode"], "area_fraction": fraction, "score_mean": current.score_mean, "score_std": current.score_std, "fraction_score_gt_0p98": current.fraction_gt_098, "mAP": ranking_map, "AUROC": ranking_auc, "PixelBestDice": best, **calibrated})
                completed_image_passes += 1
                write_json(PROGRESS_OUT, {"status": "running", "fold": test_fold, "phase": phase, "image_id": image_id, "completed_image_passes": completed_image_passes, "total_image_passes": total_image_passes, "completed_queries": completed_queries, "total_queries": len(rows) * (2 if args.fold < 0 else 1)})
                print(f"token_geometry fold={test_fold} phase={phase} image={image_id}", flush=True)
        for (name, class_id), grids in calibration_grids.items():
            calibration_source = grids if grids else calibration_global_grids[name]
            if not calibration_source:
                raise RuntimeError(f"no calibration grids for candidate={name}")
            mean_grid = mean_dice_grid(calibration_source)
            best_index = int(np.argmax(mean_grid))
            fraction_rows.append({"candidate": name, "test_fold": test_fold, "calibration_fold": calibration_fold, "target_class": class_id, "n_calibration_queries": len(grids), "area_fraction": float(FRACTIONS[best_index]), "calibration_PixelDice": float(mean_grid[best_index])})
        if args.fold < 0:
            oof_path, fractions_path, meta_path = checkpoint_paths(test_fold)
            write_csv(oof_path, oof_rows[fold_oof_start:])
            write_csv(fractions_path, fraction_rows[fold_fractions_start:])
            write_json(meta_path, {"split_audit": fold_audit, "image_passes": 2 * len(fold_audit["test_images"]), "query_passes": 2 * sum(len(by_image[image_id]) for image_id in fold_audit["test_images"]), "oof_rows": len(oof_rows) - fold_oof_start})
        image_data_cache.clear()
        del transform
        gc.collect()
        torch.cuda.empty_cache()

    if args.fold >= 0:
        write_json(PROGRESS_OUT, {"status": "complete", "completed_image_passes": completed_image_passes, "total_image_passes": total_image_passes, "completed_queries": completed_queries, "total_queries": completed_queries})
        print(json.dumps({"fold": args.fold, "rows": len(oof_rows)}, ensure_ascii=False))
        return
    summary = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in oof_rows:
        groups[(str(row["candidate"]), "all")].append(row)
        groups[(str(row["candidate"]), str(row["prompt_quality"]))].append(row)
    for (candidate, quality), group in sorted(groups.items()):
        summary.append({"candidate": candidate, "prompt_quality": quality, "n": len(group), **{key: float(np.mean([float(row[key]) for row in group])) for key in ("score_mean", "score_std", "fraction_score_gt_0p98", "mAP", "AUROC", "PixelBestDice", "PixelDice", "Pixel_mIoU", "PixelPrecision", "PixelRecall")}})
    write_csv(PREDICTIONS_OUT, oof_rows)
    write_csv(FRACTIONS_OUT, fraction_rows)
    write_csv(SUMMARY_OUT, summary)
    overall = [row for row in summary if row["prompt_quality"] == "all"]
    baseline = next(row for row in overall if row["candidate"] == "enhanced_raw")
    eligible = [row["candidate"] for row in overall if row["PixelDice"] >= baseline["PixelDice"] + 0.01 and row["mAP"] >= baseline["mAP"] - 0.005 and row["AUROC"] >= baseline["AUROC"] - 0.005]
    finite_metrics = all(np.isfinite(float(row[key])) for row in oof_rows for key in ("score_mean", "score_std", "fraction_score_gt_0p98", "mAP", "AUROC", "PixelBestDice", "PixelDice", "Pixel_mIoU"))
    validation = {"passed": len(oof_rows) == 3711 * len(specs) and finite_metrics, "queries": 3711, "candidates": [name for name, _, _ in specs], "split_audit": split_audit, "projection_gate_candidates": eligible, "errors": []}
    if not validation["passed"]:
        validation["errors"].append(f"expected {3711 * len(specs)} OOF rows, got {len(oof_rows)}")
    if not finite_metrics:
        validation["errors"].append("non-finite metric values found")
    write_json(VALIDATION_OUT, validation)
    lines = ["# Token Geometry Ablation", "", "Strict 12/4/4 WSI-level direct score calibration using classwise top-area fraction.", "", "| candidate | Pixel Dice | mAP | AUROC | score std | frac(score>0.98) |", "|---|---:|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {row['candidate']} | {row['PixelDice']:.4f} | {row['mAP']:.4f} | {row['AUROC']:.4f} | {row['score_std']:.4f} | {row['fraction_score_gt_0p98']:.4f} |")
    lines.extend(["", f"Projection gate candidates: {', '.join(eligible) if eligible else 'none'}"])
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(PROGRESS_OUT, {"status": "complete", "completed_image_passes": completed_image_passes, "total_image_passes": total_image_passes, "completed_queries": completed_queries, "total_queries": len(rows) * 2})
    print(json.dumps({"validation": validation, "projection_gate_candidates": eligible}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(PROGRESS_OUT, {"status": "failed", "traceback": traceback.format_exc()})
        raise
