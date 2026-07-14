from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import PRET_DIR, read_csv, write_json_atomic


NUMERIC_FIELDS = (
    "average_precision",
    "auroc",
    "precision_at_top5_area",
    "fixed_threshold_dice",
    "top1_area_dice",
    "top5_area_dice",
    "top10_area_dice",
    "top20_area_dice",
    "best_area_dice",
    "best_area_ratio",
    "calibrated_top18_area_dice",
    "calibrated_percentile_90_dice",
    "otsu_dice",
    "mean_std_0p5_dice",
    "mean_std_1p0_dice",
    "percentile_85_dice",
    "percentile_90_dice",
    "percentile_95_dice",
    "prompt_relative_margin_0p05_dice",
    "prompt_relative_margin_0p10_dice",
    "cc_top20_keep3_dice",
    "prompt_target_area_fraction",
    "prompt_valid_area_fraction",
)


def row_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else default


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [row_float(row, key) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair PRET eval by-class and validation files from metrics CSV.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--expected_compute_jobs", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / PRET_DIR
    metrics_path = output_dir / "pret_metrics.csv"
    summary_path = output_dir / "pret_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    rows = read_csv(metrics_path)
    if not rows:
        raise RuntimeError(f"empty metrics CSV: {metrics_path}")

    finite = True
    for row in rows:
        for key in NUMERIC_FIELDS:
            if key in row and row[key] != "":
                finite = finite and bool(np.isfinite(float(row[key])))

    groups: dict[tuple[str, str, float, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["variant"],
                row.get("baseline", "none"),
                row_float(row, "smoothing_alpha"),
                row["prompt_source"],
                row["scope"],
                int(row["class_id"]),
            )
        ].append(row)

    by_class = []
    for key, subset in sorted(groups.items()):
        variant, baseline, smoothing_alpha, prompt_source, scope, class_id = key
        by_class.append(
            {
                "variant": variant,
                "baseline": baseline,
                "smoothing_alpha": smoothing_alpha,
                "prompt_source": prompt_source,
                "scope": scope,
                "class_id": class_id,
                "queries": len(subset),
                "mean_average_precision": mean(subset, "average_precision"),
                "mean_auroc": mean(subset, "auroc"),
                "mean_precision_at_top5_area": mean(subset, "precision_at_top5_area"),
                "mean_fixed_threshold_dice": mean(subset, "fixed_threshold_dice"),
                "mean_top10_area_dice": mean(subset, "top10_area_dice"),
                "mean_percentile_90_dice": mean(subset, "percentile_90_dice"),
                "mean_otsu_dice": mean(subset, "otsu_dice"),
                "mean_cc_top20_keep3_dice": mean(subset, "cc_top20_keep3_dice"),
                "mean_calibrated_top18_area_dice": mean(subset, "calibrated_top18_area_dice"),
                "mean_best_area_dice": mean(subset, "best_area_dice"),
                "acellular_focus": class_id in {3, 9, 10},
            }
        )

    write_json_atomic(output_dir / "pret_by_class.json", by_class)
    observed_expanded_jobs = sorted({(row["variant"], row["image_id"]) for row in rows})
    observed_compute_jobs = sorted(
        {
            (row["variant"], row["image_id"])
            for row in rows
            if row.get("baseline", "none") == "none" and "__" not in row["variant"]
        }
    )
    validation = {
        "passed": bool(finite and rows and by_class and summary_path.exists()),
        "results": len(rows),
        "observed_compute_jobs": len(observed_compute_jobs),
        "observed_expanded_jobs": len(observed_expanded_jobs),
        "expected_compute_jobs": args.expected_compute_jobs,
        "expected_compute_jobs_matched": bool(
            args.expected_compute_jobs == 0 or len(observed_compute_jobs) == args.expected_compute_jobs
        ),
        "variants": sorted({row["variant"] for row in rows}),
        "baselines": sorted({row.get("baseline", "none") for row in rows}),
        "smoothing_alphas": sorted({row_float(row, "smoothing_alpha") for row in rows}),
        "prompt_sources": sorted({row["prompt_source"] for row in rows}),
        "scopes": sorted({row["scope"] for row in rows}),
        "numeric_values_finite": finite,
        "repaired_from_metrics_csv": True,
        "note": "Eval log completed, but final validation/by-class files were missing after interruption; rebuilt from pret_metrics.csv.",
    }
    write_json_atomic(output_dir / "pret_eval_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"PRET validation repair failed: {validation}")
    print(validation)


if __name__ == "__main__":
    main()
