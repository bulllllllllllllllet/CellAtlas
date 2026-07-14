from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import PRET_DIR, prompt_purity_bin, read_csv, write_json_atomic


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(np.mean(values)) if values else 0.0


def summarize(rows: list[dict[str, str]], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        item.update(
            {
                "queries": len(subset),
                "mean_average_precision": mean(subset, "average_precision"),
                "mean_auroc": mean(subset, "auroc"),
                "mean_precision_at_top5_area": mean(subset, "precision_at_top5_area"),
                "mean_top5_area_dice": mean(subset, "top5_area_dice"),
                "mean_top10_area_dice": mean(subset, "top10_area_dice"),
                "mean_top20_area_dice": mean(subset, "top20_area_dice"),
                "mean_percentile_90_dice": mean(subset, "percentile_90_dice"),
                "mean_otsu_dice": mean(subset, "otsu_dice"),
                "mean_cc_top20_keep3_dice": mean(subset, "cc_top20_keep3_dice"),
                "mean_calibrated_top18_area_dice": mean(subset, "calibrated_top18_area_dice"),
                "mean_best_area_dice": mean(subset, "best_area_dice"),
                "mean_best_area_ratio": mean(subset, "best_area_ratio"),
                "mean_prompt_target_area_fraction": mean(subset, "prompt_target_area_fraction"),
            }
        )
        output.append(item)
    return output


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
    parser = argparse.ArgumentParser(description="Analyze PRET superpixel benchmark outputs.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--primary_variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--prompt_source", default="realistic_box")
    parser.add_argument("--scope", default="exclude_prompt_region")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / PRET_DIR
    rows = read_csv(output_dir / "pret_metrics.csv")
    for row in rows:
        row["prompt_purity_bin"] = prompt_purity_bin(float(row.get("prompt_target_area_fraction", 0.0)))

    purity_rows = summarize(
        rows,
        ("variant", "baseline", "smoothing_alpha", "prompt_source", "scope", "prompt_purity_bin"),
    )
    class_rows = summarize(
        rows,
        ("variant", "baseline", "smoothing_alpha", "prompt_source", "scope", "class_id"),
    )
    write_json_atomic(output_dir / "pret_purity_strata.json", purity_rows)
    write_json_atomic(output_dir / "pret_class_focus.json", class_rows)

    primary = [
        row for row in rows
        if row["variant"] == args.primary_variant
        and row["baseline"] == "none"
        and float(row["smoothing_alpha"]) == 0.0
        and row["prompt_source"] == args.prompt_source
        and row["scope"] == args.scope
    ]
    primary_by_purity = summarize(primary, ("prompt_purity_bin",))
    primary_by_class = summarize(primary, ("class_id",))
    variant_summary = summarize(
        [
            row for row in rows
            if row["baseline"] == "none"
            and float(row["smoothing_alpha"]) == 0.0
            and row["prompt_source"] == args.prompt_source
            and row["scope"] == args.scope
        ],
        ("variant",),
    )
    variant_summary.sort(key=lambda row: row["mean_average_precision"], reverse=True)

    lines = [
        "# PRET Superpixel Analysis",
        "",
        f"Primary setting: `{args.primary_variant}`, `{args.prompt_source}`, `{args.scope}`.",
        "",
        "## Variant Summary",
        "",
        *markdown_table(
            variant_summary,
            [
                ("Variant", "variant"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("AUROC", "mean_auroc"),
                ("P@top5", "mean_precision_at_top5_area"),
                ("Top10 Dice", "mean_top10_area_dice"),
                ("P90 Dice", "mean_percentile_90_dice"),
                ("Otsu Dice", "mean_otsu_dice"),
                ("CC Dice", "mean_cc_top20_keep3_dice"),
                ("Top18 Cal", "mean_calibrated_top18_area_dice"),
                ("BestDice", "mean_best_area_dice"),
                ("BestArea", "mean_best_area_ratio"),
            ],
        ),
        "",
        "## Primary By Prompt Purity",
        "",
        *markdown_table(
            primary_by_purity,
            [
                ("Purity Bin", "prompt_purity_bin"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("Top10 Dice", "mean_top10_area_dice"),
                ("P90 Dice", "mean_percentile_90_dice"),
                ("CC Dice", "mean_cc_top20_keep3_dice"),
                ("BestDice", "mean_best_area_dice"),
                ("PromptTarget", "mean_prompt_target_area_fraction"),
            ],
        ),
        "",
        "## Primary By Class",
        "",
        *markdown_table(
            primary_by_class,
            [
                ("Class", "class_id"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("P@top5", "mean_precision_at_top5_area"),
                ("Top10 Dice", "mean_top10_area_dice"),
                ("P90 Dice", "mean_percentile_90_dice"),
                ("Top18 Cal", "mean_calibrated_top18_area_dice"),
                ("BestDice", "mean_best_area_dice"),
            ],
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `BestDice` is an oracle upper bound over fixed area ratios; it is not a deployable threshold.",
        "- `P90/Otsu/CC` Dice are deployable thresholds because they use score statistics and superpixel topology, not candidate GT.",
        "- `Top18 Cal` is a calibrated engineering threshold derived from the 4096 BestArea observation, not a fully unsupervised result.",
        "- `exclude_prompt_region` measures generalization outside the user-selected prompt.",
        "- `include_prompt_region` is closer to interactive mask behavior because prompted positives are part of the final mask.",
    ]
    report = "\n".join(lines) + "\n"
    (output_dir / "PRET_SUPERPIXEL_ANALYSIS.md").write_text(report, encoding="utf-8")
    print(output_dir / "PRET_SUPERPIXEL_ANALYSIS.md")


if __name__ == "__main__":
    main()
