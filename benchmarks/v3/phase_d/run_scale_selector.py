# 作用：构建无 GT 泄漏的多尺度选择数据集，并进行 WSI-level 五折尺度选择器训练与评估。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_c.run_multiscale_baseline import parse_boxes, read_csv

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST = PRET_ROOT / "data_manifest_v3.csv"
PROMPTS = PRET_ROOT / "prompt_tasks" / "all_prompt_tasks.csv"
METRICS = PRET_ROOT / "evaluations" / "multiscale_baseline_metrics.csv"
PIXEL_METRICS = PRET_ROOT / "evaluations" / "multiscale_pixel_metrics.csv"
SCORES = PRET_ROOT / "evaluations" / "query_scale_scores"
OUT = PRET_ROOT / "scale_selection"
PHASE_DIR = Path(__file__).resolve().parent
SCALES = ("small", "medium", "large")
SCALE_TO_ID = {scale: index for index, scale in enumerate(SCALES)}
TIE_ORDER = {"medium": 0, "small": 1, "large": 2}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def score_features(path: Path) -> dict[str, float]:
    with np.load(path) as data:
        area = data["area"].astype(np.float64)
        # Inference has no GT-derived valid_fraction; use all Phase A tissue superpixels.
        values: dict[str, float] = {"valid_area": float(area.sum())}
        for name in ("score_pos", "score_neg", "score_final"):
            score = data[name].astype(np.float64)
            for suffix, value in (("mean", score.mean()), ("std", score.std()), ("p10", np.percentile(score, 10)), ("p50", np.percentile(score, 50)), ("p90", np.percentile(score, 90)), ("p99", np.percentile(score, 99)), ("max", score.max())):
                values[f"{name}_{suffix}"] = float(value)
            values[f"{name}_gap_p99_p90"] = float(np.percentile(score, 99) - np.percentile(score, 90))
            values[f"{name}_gap_p90_p50"] = float(np.percentile(score, 90) - np.percentile(score, 50))
    return values


def choose_label(group: pd.DataFrame, metric: str) -> tuple[str, float]:
    ordered = group.sort_values([metric, "tie_order"], ascending=[False, True])
    values = ordered[metric].to_numpy(dtype=float)
    return str(ordered.iloc[0]["scale"]), float(values[0] - values[1])


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(MANIFEST).set_index("image_id")
    prompts = pd.read_csv(PROMPTS).set_index("query_id")
    retrieval = pd.read_csv(METRICS)
    pixel = pd.read_csv(PIXEL_METRICS)
    metrics = retrieval.merge(
        pixel[["query_id", "image_id", "target_class", "scale", "status", "PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice"]],
        on=["query_id", "image_id", "target_class", "scale"], suffixes=("", "_pixel"), validate="one_to_one",
    )
    metrics = metrics[metrics.status.eq("ok") & metrics.status_pixel.eq("ok")].copy()
    metrics["tie_order"] = metrics.scale.map(TIE_ORDER)
    complete = metrics.groupby("query_id").filter(lambda x: len(x) == 3)
    records: list[dict[str, object]] = []
    for query_id, group in complete.groupby("query_id", sort=True):
        prompt = prompts.loc[query_id]
        image_id = str(prompt.image_id)
        positive_boxes = parse_boxes(prompt.positive_boxes)
        x0 = min(box[0] for box in positive_boxes)
        y0 = min(box[1] for box in positive_boxes)
        x1 = max(box[2] for box in positive_boxes)
        y1 = max(box[3] for box in positive_boxes)
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        row: dict[str, object] = {
            "query_id": query_id, "image_id": image_id, "fold": int(manifest.loc[image_id, "fold"]),
            "prompt_log_area": float(np.log1p(width * height)), "prompt_log_width": float(np.log1p(width)),
            "prompt_log_height": float(np.log1p(height)), "prompt_aspect_ratio": float(width / height),
            "positive_prompt_count": int(prompt.positive_segment_count), "negative_prompt_count": int(prompt.negative_segment_count),
            "prompt_quality_eval": str(prompt.prompt_quality), "target_class_eval": int(prompt.target_class),
        }
        formal_label, formal_margin = choose_label(group, "PixelDice_classwise_toparea")
        best_label, best_margin = choose_label(group, "PixelBestDice")
        row.update({"best_scale_formal_dice": formal_label, "best_scale_bestdice": best_label, "formal_margin": formal_margin, "bestdice_margin": best_margin})
        for _, metric in group.iterrows():
            scale = str(metric.scale)
            prefix = f"{scale}_"
            score = score_features(SCORES / f"{query_id}_{scale}.npz")
            row.update({prefix + key: value for key, value in score.items()})
            row[prefix + "candidate_segments"] = int(metric.candidate_segments)
            row[prefix + "positive_projected_segments"] = int(metric.positive_prompt_segments)
            row[prefix + "negative_projected_segments"] = int(metric.negative_prompt_segments)
            for metric_name in ("mAP", "AUROC", "PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice"):
                row[prefix + "eval_" + metric_name] = float(metric[metric_name])
        mean_area = np.mean([float(row[f"{scale}_valid_area"]) for scale in SCALES])
        row["prompt_area_fraction"] = float((width * height) / max(mean_area, 1.0))
        records.append(row)
        if len(records) % 500 == 0:
            print(f"scale_selector_dataset {len(records)}/{complete.query_id.nunique()}", flush=True)
    dataset = pd.DataFrame(records)
    excluded = {"query_id", "image_id", "fold", "prompt_quality_eval", "target_class_eval", "best_scale_formal_dice", "best_scale_bestdice", "formal_margin", "bestdice_margin"}
    excluded.update(column for column in dataset if "_eval_" in column)
    features = [column for column in dataset.columns if column not in excluded]
    return dataset, features


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded = {
        "query_id", "image_id", "fold", "prompt_quality_eval", "target_class_eval",
        "best_scale_formal_dice", "best_scale_bestdice", "formal_margin", "bestdice_margin",
    }
    excluded.update(column for column in dataset if "_eval_" in column)
    return [column for column in dataset.columns if column not in excluded]


def model_factory(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "gbdt":
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=seed))])


def metrics_for_choice(row: pd.Series, scale: str) -> dict[str, float]:
    return {name: float(row[f"{scale}_eval_{name}"]) for name in ("mAP", "AUROC", "PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice")}


def add_prediction(rows: list[dict[str, object]], data: pd.DataFrame, method: str, label_name: str, label_column: str | None, prediction: np.ndarray, fold: int) -> None:
    for (_, source), scale in zip(data.iterrows(), prediction, strict=True):
        chosen = str(scale)
        item = {"query_id": source.query_id, "image_id": source.image_id, "fold": fold, "method": method, "label_target": label_name, "predicted_scale": chosen, "true_scale": source[label_column] if label_column else "", "prompt_quality_eval": source.prompt_quality_eval, "target_class_eval": int(source.target_class_eval), **metrics_for_choice(source, chosen)}
        rows.append(item)


def train(dataset: pd.DataFrame, features: list[str], margin: float) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    final_candidates: list[tuple[str, str]] = []
    labels = (("formal", "best_scale_formal_dice", "formal_margin"), ("bestdice", "best_scale_bestdice", "bestdice_margin"))
    for fold in sorted(dataset.fold.unique()):
        test = dataset[dataset.fold.eq(fold)].copy()
        train_all = dataset[~dataset.fold.eq(fold)].copy()
        thresholds = train_all.prompt_area_fraction.quantile([1 / 3, 2 / 3]).to_numpy()
        manual = np.where(test.prompt_area_fraction <= thresholds[0], "small", np.where(test.prompt_area_fraction <= thresholds[1], "medium", "large"))
        for fixed in SCALES:
            add_prediction(prediction_rows, test, f"fixed_{fixed}", "", None, np.repeat(fixed, len(test)), int(fold))
        add_prediction(prediction_rows, test, "manual_area_rule", "", None, manual, int(fold))
        for target_name, label_column, margin_column in labels:
            train = train_all[train_all[margin_column].ge(margin)].copy()
            y = train[label_column].to_numpy()
            for model_name in ("rf", "gbdt", "mlp"):
                model = model_factory(model_name, 2026 + int(fold))
                if model_name == "mlp":
                    model.fit(train[features], np.asarray([SCALE_TO_ID[scale] for scale in y], dtype=np.int64))
                    prediction = np.asarray(SCALES)[model.predict(test[features]).astype(np.int64)]
                else:
                    model.fit(train[features], y)
                    prediction = model.predict(test[features])
                add_prediction(prediction_rows, test, f"learned_{model_name}", target_name, label_column, prediction, int(fold))
                final_candidates.append((model_name, target_name))
    frame = pd.DataFrame(prediction_rows)
    summary: list[dict[str, object]] = []
    for (method, label), group in frame.groupby(["method", "label_target"], dropna=False):
        learned = method.startswith("learned_")
        summary.append({"method": method, "label_target": label, "n": len(group), "scale_accuracy": float(accuracy_score(group.true_scale, group.predicted_scale)) if learned else "", "macro_f1": float(f1_score(group.true_scale, group.predicted_scale, average="macro")) if learned else "", **{name: float(group[name].mean()) for name in ("mAP", "AUROC", "PixelDice_classwise_toparea", "Pixel_mIoU", "PixelBestDice")}})
    summary_frame = pd.DataFrame(summary)
    learned = summary_frame[summary_frame.method.str.startswith("learned_")].sort_values(["PixelDice_classwise_toparea", "Pixel_mIoU", "macro_f1"], ascending=False)
    winner = learned.iloc[0]
    model_name, label_target = str(winner.method).removeprefix("learned_"), str(winner.label_target)
    label_column = "best_scale_formal_dice" if label_target == "formal" else "best_scale_bestdice"
    margin_column = "formal_margin" if label_target == "formal" else "bestdice_margin"
    final_train = dataset[dataset[margin_column].ge(margin)]
    final_model = model_factory(model_name, 2026)
    if model_name == "mlp":
        final_model.fit(final_train[features], np.asarray([SCALE_TO_ID[scale] for scale in final_train[label_column]], dtype=np.int64))
    else:
        final_model.fit(final_train[features], final_train[label_column])
    payload = {"model": final_model, "features": features, "label_target": label_target, "min_label_margin": margin, "token": "tokens_image_cell_reg_texture_cellstats.npy", "class_id_to_scale": dict(enumerate(SCALES)) if model_name == "mlp" else None, "winner": winner.to_dict()}
    return prediction_rows, summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_label_margin", type=float, default=0.01)
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--train_only", action="store_true", help="复用已写入的数据集，仅执行五折训练。")
    args = parser.parse_args()
    if args.build_only and args.train_only:
        raise ValueError("--build_only and --train_only cannot be used together")
    if args.train_only:
        dataset = pd.read_csv(OUT / "scale_selector_dataset.csv")
        features = feature_columns(dataset)
    else:
        dataset, features = build_dataset()
    if dataset.query_id.nunique() != len(dataset) or len(dataset) != 3704 or not np.isfinite(dataset[features].to_numpy(dtype=float)).all():
        raise RuntimeError("scale selector dataset validation failed")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "scale_selector_dataset.csv", dataset.to_dict("records"))
    validation = {"passed": True, "queries": len(dataset), "images": int(dataset.image_id.nunique()), "fold_counts": dataset.groupby("fold").image_id.nunique().to_dict(), "features": features, "min_label_margin": args.min_label_margin}
    if args.build_only:
        (OUT / "scale_selector_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        return
    predictions, summary, model = train(dataset, features, args.min_label_margin)
    write_csv(OUT / "scale_selector_oof_predictions.csv", predictions)
    write_csv(OUT / "scale_selector_results.csv", summary)
    joblib.dump(model, OUT / "scale_selector_model.pkl")
    validation["winner"] = model["winner"]
    validation["oof_predictions"] = len(predictions)
    (OUT / "scale_selector_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    lines = ["# Phase D Prompt Scale Selector", "", f"- queries: {len(dataset)}", f"- feature count: {len(features)}", "- label/evaluation: raw-GT pixel-level Dice and mIoU; input features exclude GT-derived valid_fraction.", f"- winner: {model['winner']}", "", "| method | label | Pixel Dice | Pixel mIoU | mAP | AUROC | accuracy | macro F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['label_target']} | {row['PixelDice_classwise_toparea']:.4f} | {row['Pixel_mIoU']:.4f} | {row['mAP']:.4f} | {row['AUROC']:.4f} | {row['scale_accuracy'] if row['scale_accuracy'] != '' else ''} | {row['macro_f1'] if row['macro_f1'] != '' else ''} |")
    (PHASE_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
