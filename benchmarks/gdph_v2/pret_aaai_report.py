from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.pret_utils import PRET_DIR, read_csv


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_csv_optional(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.is_file() else []


def _read_json_optional(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [_float(row, key) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else 0.0


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
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


def _select_summary(
    rows: list[dict[str, str]],
    variant: str | None = None,
    prototype: str | None = None,
    threshold: str | None = None,
    scope: str = "exclude_prompt_region",
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if variant is not None and row.get("variant") != variant:
            continue
        if prototype is not None and row.get("prototype_protocol") != prototype:
            continue
        if threshold is not None and row.get("threshold_protocol") != threshold:
            continue
        if row.get("scope") != scope:
            continue
        output.append(row)
    return output


def _best_by(rows: list[dict[str, str]], group_key: str, metric: str) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    output = []
    for key, subset in grouped.items():
        best = max(subset, key=lambda row: _float(row, metric))
        item = dict(best)
        item[group_key] = key
        output.append(item)
    return sorted(output, key=lambda row: row[group_key])


def _sam_status_lines(pret_dir: Path, model_name: str) -> list[str]:
    status = _read_json_optional(pret_dir / f"{model_name}_baseline_status.json")
    summary = _read_csv_optional(pret_dir / f"{model_name}_baseline_summary.csv")
    if summary:
        overall = summary[0]
        return _markdown_table(
            [
                {
                    "Model": model_name,
                    "Queries": int(float(overall["queries"])),
                    "Dice": _float(overall, "mean_dice"),
                    "mIoU": _float(overall, "mean_iou"),
                    "BF1@5": _float(overall, "mean_boundary_f1_5px"),
                    "BF1@10": _float(overall, "mean_boundary_f1_10px"),
                }
            ],
            [("Model", "Model"), ("Queries", "Queries"), ("Dice", "Dice"), ("mIoU", "mIoU"), ("BF1@5", "BF1@5"), ("BF1@10", "BF1@10")],
        )
    if status:
        return [f"- `{model_name}`: `{status.get('status', 'unknown')}` ({status.get('checkpoint', status.get('error', ''))})"]
    return [f"- `{model_name}`: not run"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write unified AAAI PRET-superpixel baseline report.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--primary_variant", default="image_cell_reg_cellw0p5")
    parser.add_argument("--primary_prototype", default="median")
    parser.add_argument("--primary_threshold", default="p90")
    args = parser.parse_args()

    root = Path(args.output_root)
    pret_dir = root / PRET_DIR
    summary_rows = _read_csv_optional(pret_dir / "pret_aaai_summary.csv")
    class_rows = _read_csv_optional(pret_dir / "pret_aaai_by_class.csv")
    if not summary_rows:
        raise RuntimeError(f"missing AAAI summary: {pret_dir / 'pret_aaai_summary.csv'}")

    primary_rows = _select_summary(summary_rows, prototype=args.primary_prototype, threshold=args.primary_threshold)
    variant_rows = [
        row for row in primary_rows
        if row["variant"] in {args.primary_variant, "patch_only", "image_only", "cell_reg"}
    ]
    threshold_rows = _select_summary(summary_rows, variant=args.primary_variant, prototype=args.primary_prototype)
    prototype_rows = _select_summary(summary_rows, variant=args.primary_variant, threshold=args.primary_threshold)
    best_threshold_rows = _best_by(threshold_rows, "threshold_protocol", "mean_dice")
    best_proto_rows = _best_by(prototype_rows, "prototype_protocol", "mean_dice")
    primary_class = [
        row for row in class_rows
        if row["variant"] == args.primary_variant
        and row["prototype_protocol"] == args.primary_prototype
        and row["threshold_protocol"] == args.primary_threshold
        and row["scope"] == "exclude_prompt_region"
    ]

    report = [
        "# PRET-superpixel AAAI baseline report",
        "",
        "## Protocol",
        f"- Source: `{args.output_root}`",
        "- Main setting: full10x auto-physical superpixels, `realistic_box` prompt, target-vs-rest evaluation.",
        "- `mAP` is threshold-free retrieval AP. `Dice`, `mIoU`, and Boundary F1 are binary mask metrics after thresholding.",
        "- PRET Boundary F1 is computed on the superpixel adjacency graph as a scalable boundary proxy: BF1@5 = 1-hop tolerance, BF1@10 = 2-hop tolerance.",
        "- `mIoU` here is binary target-vs-rest IoU averaged across queries/classes, not a 12-class dense decoder mIoU.",
        "- SAM/MedSAM, when available, are local box-prompt segmentation baselines and do not produce full-slide retrieval mAP.",
        "",
        "## Core Baselines",
        *_markdown_table(
            variant_rows,
            [
                ("Variant", "variant"),
                ("Proto", "prototype_protocol"),
                ("Threshold", "threshold_protocol"),
                ("Scope", "scope"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("AUROC", "mean_auroc"),
                ("Dice", "mean_dice"),
                ("mIoU", "mean_binary_miou"),
                ("BF1@5", "mean_boundary_f1_5px"),
                ("BF1@10", "mean_boundary_f1_10px"),
            ],
        ),
        "",
        "## Threshold Calibration",
        *_markdown_table(
            best_threshold_rows,
            [
                ("Threshold", "threshold_protocol"),
                ("Scope", "scope"),
                ("Queries", "queries"),
                ("Dice", "mean_dice"),
                ("mIoU", "mean_binary_miou"),
                ("BF1@5", "mean_boundary_f1_5px"),
                ("PredArea", "mean_pred_area_fraction"),
            ],
        ),
        "",
        "## Prototype Refinement",
        *_markdown_table(
            best_proto_rows,
            [
                ("Prototype", "prototype_protocol"),
                ("Threshold", "threshold_protocol"),
                ("Scope", "scope"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("Dice", "mean_dice"),
                ("mIoU", "mean_binary_miou"),
                ("BF1@5", "mean_boundary_f1_5px"),
            ],
        ),
        "",
        "## SAM / MedSAM Box-Prompt Baselines",
        *_sam_status_lines(pret_dir, "sam_vit_b"),
        "",
        *_sam_status_lines(pret_dir, "medsam_vit_b"),
        "",
        "## Per-Class Primary Result",
        *_markdown_table(
            primary_class,
            [
                ("Class", "class_name"),
                ("Queries", "queries"),
                ("mAP", "mean_average_precision"),
                ("Dice", "mean_dice"),
                ("mIoU", "mean_iou"),
                ("BF1@5", "mean_boundary_f1_5px"),
                ("BF1@10", "mean_boundary_f1_10px"),
            ],
        ),
        "",
        "## Visualizations",
        f"- Folder: `{pret_dir / 'aaai_visualizations'}`",
        "- Recommended cases: best/median/worst by class for the primary variant; each case should keep eight separate PNG panels and a `summary.json`.",
        "",
        "## Files",
        f"- Metrics: `{pret_dir / 'pret_aaai_metrics.csv'}`",
        f"- Summary: `{pret_dir / 'pret_aaai_summary.csv'}`",
        f"- Per-class: `{pret_dir / 'pret_aaai_by_class.csv'}`",
    ]
    report_path = pret_dir / "PRET_SUPERPIXEL_AAAI_BASELINE.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
