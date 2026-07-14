from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.pret_utils import PRET_DIR, prompt_purity_bin, read_csv, write_csv_atomic, write_json_atomic


DEFAULT_FIELDS = (
    "average_precision",
    "auroc",
    "precision_at_top5_area",
    "top10_area_dice",
    "top18_area_dice",
    "top20_area_dice",
    "percentile_90_dice",
    "otsu_dice",
    "calibrated_top18_area_dice",
    "best_area_dice",
    "best_area_ratio",
)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"run must be NAME=OUTPUT_ROOT, got: {value}")
    name, path = value.split("=", 1)
    return name, Path(path)


def row_key(row: dict[str, str]) -> tuple[str, int]:
    return row["query_id"], int(row.get("shot", 0))


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        seen.setdefault(row_key(row), row)
    return [seen[key] for key in sorted(seen)]


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else 0.0


def summarize_rows(rows: list[dict[str, str]], random_rows: list[dict[str, str]] | None = None) -> dict:
    output = {"queries": len(rows)}
    for field in DEFAULT_FIELDS:
        output[field] = mean(rows, field) if any(row.get(field, "") != "" for row in rows) else 0.0
    output["random_average_precision"] = 0.0
    output["normalized_map_gain"] = 0.0
    if random_rows:
        random_map = mean(random_rows, "average_precision")
        output["random_average_precision"] = random_map
        method_map = output.get("average_precision", 0.0)
        output["normalized_map_gain"] = (method_map - random_map) / max(1.0 - random_map, 1e-8)
    return output


def filter_metrics(
    root: Path,
    variant: str,
    baseline: str,
    prompt_source: str,
    scope: str,
    smoothing_alpha: float,
) -> list[dict[str, str]]:
    path = root / PRET_DIR / "pret_metrics.csv"
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row.get("variant") == variant
                and row.get("baseline", "none") == baseline
                and row.get("prompt_source") == prompt_source
                and row.get("scope") == scope
                and float(row.get("smoothing_alpha", 0.0)) == smoothing_alpha
            ):
                rows.append(row)
    return rows


def class_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(int(row["class_id"]) for row in rows)


def purity_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(prompt_purity_bin(float(row.get("prompt_target_area_fraction", 0.0))) for row in rows)


def superpixel_stats(root: Path) -> dict:
    validations = []
    purity_values = []
    valid_counts = []
    positive_fraction_by_class: dict[int, list[float]] = defaultdict(list)
    for validation_path in sorted((root / PRET_DIR).glob("*/validation.json")):
        with open(validation_path, "r", encoding="utf-8") as file:
            validations.append(json.load(file))
        superpixels_csv = validation_path.parent / "superpixels.csv"
        if not superpixels_csv.exists():
            continue
        rows = read_csv(superpixels_csv)
        valid_rows = [row for row in rows if row["valid_for_retrieval"].lower() == "true"]
        valid_counts.append(len(valid_rows))
        purity_values.extend(float(row["gt_label_purity"]) for row in valid_rows)
        total_valid = max(len(valid_rows), 1)
        counts = Counter(int(row["gt_tissue_label"]) for row in valid_rows)
        for class_id, count in counts.items():
            positive_fraction_by_class[class_id].append(count / total_valid)

    def avg(key: str) -> float:
        values = [float(item[key]) for item in validations if key in item]
        return float(np.mean(values)) if values else 0.0

    return {
        "slides": len(validations),
        "mean_segments": avg("n_segments_actual"),
        "mean_segment_area_median": avg("segment_area_median"),
        "mean_empty_cell_segment_ratio": avg("empty_cell_segment_ratio"),
        "mean_gt_label_purity": float(np.mean(purity_values)) if purity_values else 0.0,
        "median_gt_label_purity": float(np.median(purity_values)) if purity_values else 0.0,
        "mean_valid_candidate_segments": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "positive_fraction_by_class": {
            str(class_id): float(np.mean(values))
            for class_id, values in sorted(positive_fraction_by_class.items())
        },
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PRET superpixel runs with all/common query summaries.")
    parser.add_argument("--runs", nargs="+", required=True, help="NAME=OUTPUT_ROOT entries")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--random_variant", default="random_token")
    parser.add_argument("--baseline", default="none")
    parser.add_argument("--prompt_source", default="realistic_box")
    parser.add_argument("--scope", default="exclude_prompt_region")
    parser.add_argument("--smoothing_alpha", type=float, default=0.0)
    args = parser.parse_args()

    runs = [parse_run(value) for value in args.runs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_run_rows = {
        name: filter_metrics(root, args.variant, args.baseline, args.prompt_source, args.scope, args.smoothing_alpha)
        for name, root in runs
    }
    raw_random_rows = {
        name: filter_metrics(root, args.random_variant, args.baseline, args.prompt_source, args.scope, args.smoothing_alpha)
        for name, root in runs
    }
    run_rows = {name: dedupe_rows(rows) for name, rows in raw_run_rows.items()}
    random_rows = {name: dedupe_rows(rows) for name, rows in raw_random_rows.items()}
    key_sets = {name: {row_key(row) for row in rows} for name, rows in run_rows.items()}
    common_keys = set.intersection(*key_sets.values()) if key_sets else set()

    summary_rows = []
    for name, _ in runs:
        rows = run_rows[name]
        random = random_rows[name]
        all_summary = summarize_rows(rows, random)
        all_summary.update(
            {
                "run": name,
                "subset": "all",
                "raw_rows": len(raw_run_rows[name]),
                "unique_query_ids": len({row["query_id"] for row in rows}),
            }
        )
        common = [row for row in rows if row_key(row) in common_keys]
        common_random = [row for row in random if row_key(row) in common_keys]
        common_summary = summarize_rows(common, common_random)
        common_summary.update(
            {
                "run": name,
                "subset": "common",
                "raw_rows": "",
                "unique_query_ids": len({row["query_id"] for row in common}),
            }
        )
        summary_rows.extend([all_summary, common_summary])

    class_rows = []
    purity_rows = []
    for name, _ in runs:
        rows = run_rows[name]
        common = [row for row in rows if row_key(row) in common_keys]
        all_counts = class_counts(rows)
        common_counts = class_counts(common)
        only_counts = class_counts([row for row in rows if row_key(row) not in common_keys])
        for class_id in sorted(set(all_counts) | set(common_counts) | set(only_counts)):
            class_rows.append(
                {
                    "run": name,
                    "class_id": class_id,
                    "all": all_counts[class_id],
                    "common": common_counts[class_id],
                    "only": only_counts[class_id],
                }
            )
        all_purity = purity_counts(rows)
        common_purity = purity_counts(common)
        only_purity = purity_counts([row for row in rows if row_key(row) not in common_keys])
        for bin_name in ["<0.5", "0.5-0.7", "0.7-0.9", ">0.9"]:
            purity_rows.append(
                {
                    "run": name,
                    "prompt_purity_bin": bin_name,
                    "all": all_purity[bin_name],
                    "common": common_purity[bin_name],
                    "only": only_purity[bin_name],
                }
            )

    superpixel_rows = []
    for name, root in runs:
        stats = superpixel_stats(root)
        stats["run"] = name
        superpixel_rows.append(stats)

    write_csv_atomic(output_dir / "pret_scale_comparison.csv", summary_rows)
    write_csv_atomic(output_dir / "pret_scale_class_comparison.csv", class_rows)
    write_csv_atomic(output_dir / "pret_scale_prompt_purity_comparison.csv", purity_rows)
    write_json_atomic(output_dir / "pret_scale_superpixel_stats.json", superpixel_rows)
    validation = {
        "passed": bool(summary_rows and common_keys),
        "variant": args.variant,
        "prompt_source": args.prompt_source,
        "scope": args.scope,
        "common_query_shot_keys": len(common_keys),
        "runs": {
            name: {
                "raw_rows": len(raw_run_rows[name]),
                "deduped_rows": len(run_rows[name]),
                "unique_query_ids": len({row["query_id"] for row in run_rows[name]}),
            }
            for name, _ in runs
        },
    }
    write_json_atomic(output_dir / "pret_scale_comparison_validation.json", validation)

    lines = [
        "# PRET Scale Comparison",
        "",
        f"Setting: `{args.variant}`, `{args.prompt_source}`, `{args.scope}`, alpha={args.smoothing_alpha}.",
        f"Common unit: de-duplicated `query_id + shot`; common count = {len(common_keys)}.",
        "",
        "## All vs Common Summary",
        "",
        *markdown_table(
            summary_rows,
            [
                ("Run", "run"),
                ("Subset", "subset"),
                ("Rows", "queries"),
                ("RawRows", "raw_rows"),
                ("UniqueQ", "unique_query_ids"),
                ("mAP", "average_precision"),
                ("AUROC", "auroc"),
                ("P@top5", "precision_at_top5_area"),
                ("Top10 Dice", "top10_area_dice"),
                ("Top20 Dice", "top20_area_dice"),
                ("BestDice", "best_area_dice"),
                ("BestArea", "best_area_ratio"),
                ("Random mAP", "random_average_precision"),
                ("NormGain", "normalized_map_gain"),
            ],
        ),
        "",
        "## Query Class Counts",
        "",
        *markdown_table(class_rows, [("Run", "run"), ("Class", "class_id"), ("All", "all"), ("Common", "common"), ("Only", "only")]),
        "",
        "## Prompt Purity Counts",
        "",
        *markdown_table(purity_rows, [("Run", "run"), ("Purity", "prompt_purity_bin"), ("All", "all"), ("Common", "common"), ("Only", "only")]),
        "",
        "## Superpixel Stats",
        "",
        *markdown_table(
            superpixel_rows,
            [
                ("Run", "run"),
                ("Slides", "slides"),
                ("MeanSeg", "mean_segments"),
                ("MedianArea", "mean_segment_area_median"),
                ("EmptyCell", "mean_empty_cell_segment_ratio"),
                ("MeanPurity", "mean_gt_label_purity"),
                ("ValidSeg", "mean_valid_candidate_segments"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Use `all` to describe each benchmark as generated at that scale.",
        "- Use `common` for fair scale comparison because it fixes the query/prompt-shot set.",
        "- `only` rows explain which extra scale-specific prompts changed benchmark difficulty.",
    ]
    report = "\n".join(lines) + "\n"
    (output_dir / "PRET_SCALE_COMPARISON.md").write_text(report, encoding="utf-8")
    print(output_dir / "PRET_SCALE_COMPARISON.md")


if __name__ == "__main__":
    main()
