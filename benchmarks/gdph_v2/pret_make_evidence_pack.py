from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.pret_eval_in_context import scores_from_prompt
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    binary_segmentation_metrics,
    parse_segment_ids,
    percentile_threshold,
    predict_top_area,
    read_csv,
    write_csv_atomic,
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


def _primary_rows(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    filtered = [
        row for row in rows
        if row["variant"] == variant
        and row["baseline"] == "none"
        and row["prompt_source"] == "realistic_box"
        and row["scope"] == "exclude_prompt_region"
        and float(row["smoothing_alpha"]) == 0.0
    ]
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in filtered:
        key = (row["query_id"], row["shot"], row["prompt_source"], row["scope"])
        by_key[key] = row
    return list(by_key.values())


def summarize_variant(rows: list[dict[str, str]], variant: str) -> dict[str, float]:
    subset = _primary_rows(rows, variant)
    return {
        "queries": float(len(subset)),
        "mAP": float(np.mean([_float(row, "average_precision") for row in subset])),
        "AUROC": float(np.mean([_float(row, "auroc") for row in subset])),
        "P@top5": float(np.mean([_float(row, "precision_at_top5_area") for row in subset])),
        "Top10Dice": float(np.mean([_float(row, "top10_area_dice") for row in subset])),
        "OtsuDice": float(np.mean([_float(row, "otsu_dice") for row in subset])),
        "Top18Cal": float(np.mean([_float(row, "calibrated_top18_area_dice") for row in subset])),
        "BestDice": float(np.mean([_float(row, "best_area_dice") for row in subset])),
    }


def write_failure_cases(rows: list[dict[str, str]], output_path: Path, variant: str, limit: int) -> list[dict[str, str]]:
    subset = _primary_rows(rows, variant)
    subset.sort(key=lambda row: (_float(row, "average_precision"), _float(row, "top10_area_dice")))
    fields = [
        "query_id", "image_id", "class_id", "class_name", "shot", "prompt_purity",
        "prompt_target_area_fraction", "average_precision", "precision_at_top5_area",
        "top10_area_dice", "otsu_dice", "calibrated_top18_area_dice", "best_area_dice",
    ]
    output = []
    for row in subset[:limit]:
        item = {field: row.get(field, "") for field in fields if field not in {"class_name"}}
        class_id = int(row["class_id"])
        item["class_name"] = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        output.append(item)
    write_csv_atomic(output_path, output)
    return output


def negative_prompt_ablation(
    root: Path,
    rows: list[dict[str, str]],
    variant: str,
    limit: int,
    negative_count: int,
) -> list[dict]:
    prompts = {
        (row["query_id"], row["prompt_source"], row["shot"]): row
        for row in read_csv(root / PRET_DIR / "prompts.csv")
    }
    candidates = [
        row for row in _primary_rows(rows, variant)
        if _float(row, "average_precision") < 0.5 or _float(row, "top10_area_dice") < 0.25
    ]
    candidates.sort(key=lambda row: (_float(row, "top10_area_dice"), _float(row, "average_precision")))
    output = []
    for metric in candidates[:limit]:
        prompt = prompts.get((metric["query_id"], metric["prompt_source"], metric["shot"]))
        if not prompt:
            continue
        image_id = metric["image_id"]
        superpixel_dir = root / PRET_DIR / image_id
        records = read_csv(superpixel_dir / "superpixels.csv")
        tokens = np.load(superpixel_dir / f"tokens_{variant}.npy", mmap_mode="r")
        positive = parse_segment_ids(prompt["positive_segments"])
        if not positive:
            continue
        labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
        valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records])
        areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
        target = labels == int(metric["class_id"])
        prompt_mask = np.zeros(len(records), dtype=bool)
        prompt_mask[positive] = True
        candidate = valid & ~prompt_mask
        if int(np.sum(candidate & target)) == 0 or int(np.sum(candidate & ~target)) == 0:
            continue
        base_scores = scores_from_prompt(np.asarray(tokens), positive, [], "negative_bank_max", 0.5)
        candidate_indices = np.flatnonzero(candidate)
        fp_pool = candidate_indices[~target[candidate_indices]]
        if len(fp_pool) == 0:
            continue
        ranked_fp = fp_pool[np.argsort(base_scores[fp_pool])[::-1]]
        negatives = ranked_fp[:negative_count].astype(int).tolist()
        corrected_scores = scores_from_prompt(np.asarray(tokens), positive, negatives, "negative_bank_max", 0.5)
        base_candidate_scores = base_scores[candidate]
        corrected_candidate_scores = corrected_scores[candidate]
        base_pred = base_candidate_scores >= percentile_threshold(base_candidate_scores, 90.0)
        corrected_pred = corrected_candidate_scores >= percentile_threshold(corrected_candidate_scores, 90.0)
        candidate_target = target[candidate]
        base_metrics = binary_segmentation_metrics(candidate_target, base_pred)
        corrected_metrics = binary_segmentation_metrics(candidate_target, corrected_pred)
        output.append(
            {
                "query_id": metric["query_id"],
                "image_id": image_id,
                "class_id": int(metric["class_id"]),
                "class_name": CLASS_NAMES[int(metric["class_id"])],
                "shot": int(metric["shot"]),
                "negative_count": len(negatives),
                "base_ap": _float(metric, "average_precision"),
                "base_p90_dice": float(base_metrics["dice"]),
                "corrected_p90_dice": float(corrected_metrics["dice"]),
                "delta_p90_dice": float(corrected_metrics["dice"] - base_metrics["dice"]),
                "base_fp": int(base_metrics["false_positive"]),
                "corrected_fp": int(corrected_metrics["false_positive"]),
                "delta_fp": int(corrected_metrics["false_positive"] - base_metrics["false_positive"]),
                "note": "oracle user-correction simulation: negatives are top-scoring non-target FP segments",
            }
        )
    return output


def markdown_table(rows: list[dict[str, str | float]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact PRET-superpixel evidence pack.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--failure_limit", type=int, default=30)
    parser.add_argument("--negative_limit", type=int, default=20)
    parser.add_argument("--negative_count", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.output_root)
    pret_dir = root / PRET_DIR
    metrics = read_csv(pret_dir / "pret_metrics.csv")
    variants = [
        "image_cell_reg_cellw0p5",
        "image_cell_reg_cellw0p25",
        "image_only",
        "cell_reg",
        "random_token",
    ]
    variant_rows = []
    for variant in variants:
        item = summarize_variant(metrics, variant)
        item["Variant"] = variant
        variant_rows.append(item)

    failure_path = pret_dir / "pret_failure_cases.csv"
    failure_rows = write_failure_cases(metrics, failure_path, args.variant, args.failure_limit)
    negative_rows = negative_prompt_ablation(root, metrics, args.variant, args.negative_limit, args.negative_count)
    negative_path = pret_dir / "pret_negative_prompt_ablation.csv"
    write_csv_atomic(negative_path, negative_rows)

    scale_report_path = root / "pret_scale_comparison" / "PRET_SCALE_COMPARISON.md"
    scale_report = scale_report_path.read_text(encoding="utf-8") if scale_report_path.is_file() else ""
    report_path = pret_dir / "PRET_EVIDENCE_PACK.md"
    report_lines = [
        "# PRET-superpixel full10x evidence pack",
        "",
        "## Core result",
        markdown_table(
            variant_rows,
            ["Variant", "queries", "mAP", "AUROC", "P@top5", "Top10Dice", "OtsuDice", "Top18Cal", "BestDice"],
        ),
        "",
        "## Interpretation",
        "- full10x uses auto-physical superpixels, matching the 4096 d64 / 8192 d128 tissue-scale token size.",
        "- The fusion token is the main result; image-only and cell-only are strong baselines but lower.",
        "- BestDice is an oracle threshold upper bound; Otsu/P90/Top18 Cal are deployable or engineering thresholds.",
        "- Negative-prompt ablation below is an oracle user-correction simulation and should not be reported as deployment performance.",
        "",
        "## Worst failure candidates",
        f"CSV: `{failure_path}`",
        markdown_table(failure_rows[:10], [
            "query_id", "class_name", "shot", "prompt_target_area_fraction",
            "average_precision", "top10_area_dice", "otsu_dice", "best_area_dice",
        ]),
        "",
        "## Oracle negative-prompt ablation",
        f"CSV: `{negative_path}`",
    ]
    if negative_rows:
        report_lines.append(
            markdown_table(negative_rows[:10], [
                "query_id", "class_name", "negative_count", "base_p90_dice",
                "corrected_p90_dice", "delta_p90_dice", "base_fp", "corrected_fp", "delta_fp",
            ])
        )
        report_lines.extend(
            [
                "",
                f"Mean delta P90 Dice: {np.mean([row['delta_p90_dice'] for row in negative_rows]):.4f}",
                f"Mean delta FP: {np.mean([row['delta_fp'] for row in negative_rows]):.2f}",
            ]
        )
    else:
        report_lines.append("No eligible negative-prompt cases were generated.")
    if scale_report:
        report_lines.extend(["", "## Scale comparison excerpt", scale_report])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "failure_cases": str(failure_path),
        "negative_prompt_ablation": str(negative_path),
        "negative_rows": len(negative_rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
