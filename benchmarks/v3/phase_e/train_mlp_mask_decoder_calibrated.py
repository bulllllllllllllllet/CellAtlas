# 作用：用严格的 12/4/4 WSI 切分训练 MLP decoder，并在独立 calibration fold 上校准全局及分类别概率阈值。

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_c.run_multiscale_baseline import read_csv
from benchmarks.v3.phase_e.train_mlp_mask_decoder import (
    Decoder,
    MANIFEST,
    METRICS,
    NUM_CLASSES,
    OUT,
    PHASE_DIR,
    SAMPLE_CACHE,
    SCALE,
    SCORE_DIR,
    TOKEN_FILE,
    ImageData,
    load_image,
    node_features,
    pixel_metric,
    predict,
    write_csv,
)

PIXEL_METRICS = OUT.parent / "evaluations" / "multiscale_pixel_metrics.csv"
PREDICTIONS_OUT = OUT / "mlp_decoder_calibrated_oof_predictions.csv"
THRESHOLDS_OUT = OUT / "mlp_decoder_thresholds.csv"
RESULTS_OUT = OUT / "mlp_decoder_calibrated_results.csv"
GROUPS_OUT = OUT / "mlp_decoder_calibrated_group_summary.csv"
MODELS_OUT = OUT / "mlp_decoder_calibrated_models.pt"
VALIDATION_OUT = OUT / "mlp_decoder_calibrated_validation.json"
REPORT_OUT = PHASE_DIR / "calibrated_report.md"
METHODS = ("fixed_0p5", "global_calibrated", "classwise_calibrated", "phase_c_fixed_small")


@dataclass
class QueryCurve:
    class_id: int
    probability_desc: np.ndarray
    cumulative_target: np.ndarray
    cumulative_pred: np.ndarray
    gt_area: float


def curve_metric(curve: QueryCurve, threshold: float) -> dict[str, float]:
    count = int(np.searchsorted(-curve.probability_desc, -threshold, side="right"))
    tp = float(curve.cumulative_target[count - 1]) if count else 0.0
    pred_area = float(curve.cumulative_pred[count - 1]) if count else 0.0
    fp, fn = pred_area - tp, curve.gt_area - tp
    denominator = 2.0 * tp + fp + fn
    union = tp + fp + fn
    return {
        "dice": 2.0 * tp / denominator if denominator else 1.0,
        "miou": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }


def make_curve(data: ImageData, class_id: int, probability: np.ndarray, valid: np.ndarray) -> QueryCurve:
    ids = np.flatnonzero(valid)
    order = np.argsort(-probability[ids], kind="stable")
    ordered = ids[order]
    return QueryCurve(
        class_id=class_id,
        probability_desc=probability[ordered].astype(np.float32),
        cumulative_target=np.cumsum(data.counts[ordered, class_id], dtype=np.float64),
        cumulative_pred=np.cumsum(data.valid_pixels[ordered], dtype=np.float64),
        gt_area=float(data.counts[:, class_id].sum()),
    )


def choose_threshold(curves: list[QueryCurve], candidates: np.ndarray) -> tuple[float, float]:
    if not curves:
        raise ValueError("cannot calibrate threshold without curves")
    scored = []
    for threshold in candidates:
        mean_dice = float(np.mean([curve_metric(curve, float(threshold))["dice"] for curve in curves]))
        scored.append((mean_dice, -abs(float(threshold) - 0.5), float(threshold)))
    best = max(scored)
    return best[2], best[0]


def train_model(
    fold: int,
    train_folds: set[int],
    sample_cache: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Decoder, np.ndarray, np.ndarray, float]:
    train_mask = np.isin(sample_cache["fold"], np.asarray(sorted(train_folds), dtype=np.int8))
    x = sample_cache["features"][train_mask].astype(np.float32)
    y = sample_cache["target"][train_mask].astype(np.float32)
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
    dataset = TensorDataset(torch.from_numpy((x - mean) / std), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = Decoder(x.shape[1]).to(device)
    positives = float(y.sum())
    pos_weight = float(np.clip((len(y) - positives) / max(positives, 1.0), 1.0, 20.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            probability = torch.sigmoid(logits)
            dice_loss = 1.0 - (2.0 * (probability * batch_y).sum() + 1.0) / (probability.sum() + batch_y.sum() + 1.0)
            loss = bce(logits, batch_y) + args.dice_weight * dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"calibrated_decoder fold={fold} epoch={epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)
    return model, mean, std, pos_weight


def infer_curve(
    row: dict[str, str],
    model: Decoder,
    mean: np.ndarray,
    std: np.ndarray,
    manifest: dict[str, str],
    cache: dict[str, ImageData],
    device: torch.device,
) -> tuple[ImageData, np.ndarray, np.ndarray, QueryCurve]:
    data = load_image(row["image_id"], manifest[row["image_id"]], cache)
    features, valid = node_features(data, SCORE_DIR / f"{row['query_id']}_{SCALE}.npz")
    probability = predict(model, features, mean, std, device)
    curve = make_curve(data, int(row["target_class"]), probability, valid)
    return data, probability, valid, curve


def aggregate(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    prefix = {
        "fixed_0p5": "Fixed05",
        "global_calibrated": "Global",
        "classwise_calibrated": "Classwise",
        "phase_c_fixed_small": "Baseline",
    }[method]
    return {
        "method": method,
        "n": len(rows),
        **{
            name: float(np.mean([float(row[f"{prefix}_{name}"]) for row in rows]))
            for name in ("PixelDice", "Pixel_mIoU", "PixelPrecision", "PixelRecall")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--calibration_fold_offset", type=int, default=1)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--fold", type=int, default=-1, help="Debug: run one outer test fold; -1 runs all folds.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("calibrated decoder requires CUDA")
    if not SAMPLE_CACHE.exists():
        raise FileNotFoundError(f"missing sample cache: {SAMPLE_CACHE}")
    candidates = np.round(np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step), 6)
    if not len(candidates) or candidates[0] <= 0 or candidates[-1] >= 1:
        raise ValueError("threshold grid must stay inside (0, 1)")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    manifest_rows = read_csv(MANIFEST)
    manifest = {row["image_id"]: row["gt_mask_path"] for row in manifest_rows}
    image_folds = {row["image_id"]: int(row["fold"]) for row in manifest_rows}
    rows = [row for row in read_csv(METRICS) if row["scale"] == SCALE and row["status"] == "ok"]
    baseline = {
        row["query_id"]: row
        for row in read_csv(PIXEL_METRICS)
        if row["scale"] == SCALE and row["status"] == "ok"
    }
    with np.load(SAMPLE_CACHE) as source:
        sample_cache = {key: source[key] for key in ("features", "target", "fold")}
    folds = sorted(set(image_folds.values()))
    requested = folds if args.fold < 0 else [args.fold]
    output: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    split_audit = []

    for test_fold in requested:
        calibration_fold = folds[(folds.index(test_fold) + args.calibration_fold_offset) % len(folds)]
        train_folds = set(folds) - {test_fold, calibration_fold}
        train_images = sorted(image for image, fold in image_folds.items() if fold in train_folds)
        calibration_images = sorted(image for image, fold in image_folds.items() if fold == calibration_fold)
        test_images = sorted(image for image, fold in image_folds.items() if fold == test_fold)
        if set(train_images) & set(calibration_images) or set(train_images) & set(test_images) or set(calibration_images) & set(test_images):
            raise RuntimeError("train/calibration/test image leakage")
        split_audit.append({"test_fold": test_fold, "calibration_fold": calibration_fold, "train_images": train_images, "calibration_images": calibration_images, "test_images": test_images})

        model, mean, std, pos_weight = train_model(test_fold, train_folds, sample_cache, args, device)
        fold_cache: dict[str, ImageData] = {}
        calibration_rows = [row for row in rows if image_folds[row["image_id"]] == calibration_fold]
        curves = []
        for index, row in enumerate(calibration_rows, start=1):
            _, _, _, curve = infer_curve(row, model, mean, std, manifest, fold_cache, device)
            curves.append(curve)
            if index % 250 == 0:
                print(f"calibrated_decoder fold={test_fold} calibration {index}/{len(calibration_rows)}", flush=True)
        global_threshold, global_score = choose_threshold(curves, candidates)
        class_thresholds: dict[int, float] = {}
        threshold_rows.append({"test_fold": test_fold, "calibration_fold": calibration_fold, "target_class": "global", "n_calibration_queries": len(curves), "threshold": global_threshold, "calibration_mean_PixelDice": global_score})
        for class_id in range(NUM_CLASSES):
            class_curves = [curve for curve in curves if curve.class_id == class_id]
            if class_curves:
                threshold, score = choose_threshold(class_curves, candidates)
            else:
                threshold, score = global_threshold, global_score
            class_thresholds[class_id] = threshold
            threshold_rows.append({"test_fold": test_fold, "calibration_fold": calibration_fold, "target_class": class_id, "n_calibration_queries": len(class_curves), "threshold": threshold, "calibration_mean_PixelDice": score})

        test_rows = [row for row in rows if image_folds[row["image_id"]] == test_fold]
        for index, row in enumerate(test_rows, start=1):
            data, probability, valid, _ = infer_curve(row, model, mean, std, manifest, fold_cache, device)
            class_id = int(row["target_class"])
            fixed = pixel_metric(data, class_id, (probability >= 0.5) & valid)
            global_metric = pixel_metric(data, class_id, (probability >= global_threshold) & valid)
            class_metric = pixel_metric(data, class_id, (probability >= class_thresholds[class_id]) & valid)
            base = baseline[row["query_id"]]
            output.append({
                "fold": test_fold, "calibration_fold": calibration_fold, "query_id": row["query_id"], "image_id": row["image_id"],
                "target_class": class_id, "prompt_quality": row["prompt_quality"], "global_threshold": global_threshold,
                "classwise_threshold": class_thresholds[class_id],
                **{f"Fixed05_{key}": fixed[source] for key, source in (("PixelDice", "dice"), ("Pixel_mIoU", "miou"), ("PixelPrecision", "precision"), ("PixelRecall", "recall"))},
                **{f"Global_{key}": global_metric[source] for key, source in (("PixelDice", "dice"), ("Pixel_mIoU", "miou"), ("PixelPrecision", "precision"), ("PixelRecall", "recall"))},
                **{f"Classwise_{key}": class_metric[source] for key, source in (("PixelDice", "dice"), ("Pixel_mIoU", "miou"), ("PixelPrecision", "precision"), ("PixelRecall", "recall"))},
                "Baseline_PixelDice": float(base["PixelDice_classwise_toparea"]), "Baseline_Pixel_mIoU": float(base["Pixel_mIoU"]),
                "Baseline_PixelPrecision": float(base["PixelPrecision"]), "Baseline_PixelRecall": float(base["PixelRecall"]), "status": "ok",
            })
            if index % 250 == 0:
                print(f"calibrated_decoder fold={test_fold} test {index}/{len(test_rows)}", flush=True)
        checkpoints.append({"test_fold": test_fold, "calibration_fold": calibration_fold, "train_folds": sorted(train_folds), "state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std, "pos_weight": pos_weight, "global_threshold": global_threshold, "class_thresholds": class_thresholds, "feature_dim": int(len(mean)), "scale": SCALE, "token_file": TOKEN_FILE})
        print(f"calibrated_decoder fold={test_fold} complete", flush=True)

    if args.fold >= 0:
        print(json.dumps({"fold": args.fold, "queries": len(output)}, ensure_ascii=False))
        return

    output.sort(key=lambda row: (int(row["fold"]), str(row["query_id"])))
    results = [aggregate(output, method) for method in METHODS]
    group_rows = []
    for method in METHODS:
        for group_type, key in (("fold", "fold"), ("target_class", "target_class"), ("prompt_quality", "prompt_quality")):
            groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in output:
                groups[str(row[key])].append(row)
            for value, members in sorted(groups.items()):
                group_rows.append({"group_type": group_type, "group_value": value, **aggregate(members, method)})
    write_csv(PREDICTIONS_OUT, output)
    write_csv(THRESHOLDS_OUT, threshold_rows)
    write_csv(RESULTS_OUT, results)
    write_csv(GROUPS_OUT, group_rows)
    torch.save({"fold_checkpoints": checkpoints, "args": vars(args)}, MODELS_OUT)

    errors = []
    if len(output) != 3711 or len({str(row["query_id"]) for row in output}) != 3711:
        errors.append("OOF predictions must contain 3711 unique queries")
    numeric = [float(row[key]) for row in output for prefix in ("Fixed05", "Global", "Classwise", "Baseline") for key in (f"{prefix}_PixelDice", f"{prefix}_Pixel_mIoU", f"{prefix}_PixelPrecision", f"{prefix}_PixelRecall")]
    if not np.isfinite(numeric).all():
        errors.append("non-finite metrics")
    validation = {"passed": not errors, "queries": len(output), "device": torch.cuda.get_device_name(0), "split_audit": split_audit, "results": results, "errors": errors}
    VALIDATION_OUT.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase E Calibrated MLP Mask Decoder", "", "Strict 12/4/4 WSI-level train/calibration/test protocol.", "", "| method | n | Pixel Dice | Pixel mIoU | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['method']} | {row['n']} | {row['PixelDice']:.4f} | {row['Pixel_mIoU']:.4f} | {row['PixelPrecision']:.4f} | {row['PixelRecall']:.4f} |")
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": validation["passed"], "queries": len(output), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
