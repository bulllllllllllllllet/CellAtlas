from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.gdph_v2.pret_utils import PRET_DIR, read_csv


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _find_summary(rows: list[dict[str, str]], protocol: str, threshold: str, scope: str = "exclude_prompt_region", negative_source: str | None = None) -> dict[str, str] | None:
    for row in rows:
        if row.get("interaction_protocol") != protocol:
            continue
        if row.get("threshold_protocol") != threshold:
            continue
        if row.get("scope") != scope:
            continue
        if negative_source is not None and row.get("negative_source") != negative_source:
            continue
        return row
    return None


def _metric_cells(row: dict[str, str] | None) -> str:
    if not row:
        return "| - | - | - | - | - | - | - |\n"
    return (
        f"| {int(float(row.get('queries', 0)))} "
        f"| {_float(row, 'mean_average_precision'):.4f} "
        f"| {_float(row, 'mean_precision_at_top5_area'):.4f} "
        f"| {_float(row, 'mean_dice'):.4f} "
        f"| {_float(row, 'mean_binary_miou'):.4f} "
        f"| {_float(row, 'mean_precision'):.4f} "
        f"| {_float(row, 'mean_recall'):.4f} |\n"
    )


def _baseline_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AAAI next experiment Markdown report.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--baseline_root", default="/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_full10x_auto_physical")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / PRET_DIR
    baseline_dir = Path(args.baseline_root) / PRET_DIR
    summary = read_csv(output_dir / "pret_aaai_next_summary.csv")
    by_class = read_csv(output_dir / "pret_aaai_next_by_class.csv") if (output_dir / "pret_aaai_next_by_class.csv").exists() else []
    supervised = _baseline_rows(output_dir / "pret_supervised_upper_bound_overall.csv")
    aaai_summary = _baseline_rows(baseline_dir / "pret_aaai_summary.csv")
    sam_summary = _baseline_rows(baseline_dir / "sam_vit_b_baseline_summary.csv")
    medsam_summary = _baseline_rows(baseline_dir / "medsam_vit_b_baseline_summary.csv")

    lines: list[str] = []
    lines.append("# PRET Superpixel AAAI Next Experiments\n\n")
    lines.append("## Main Interaction Curve\n\n")
    lines.append("Metrics are averaged over prompt queries. mIoU is binary target-vs-rest IoU, not dense 12-class decoder mIoU.\n\n")
    lines.append("| Setting | Threshold | Queries | mAP | P@top5 area | Dice | mIoU | Precision | Recall |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows_to_show = [
        ("1pos", "p90", None, "1 positive, P90"),
        ("1pos", "global_toparea_loio", None, "1 positive, global LOIO"),
        ("1pos", "prompt_adaptive_area_loio", None, "1 positive, prompt-adaptive LOIO"),
        ("1pos", "classwise_toparea_loio", None, "1 positive, classwise LOIO"),
        ("2pos", "classwise_toparea_loio", None, "2 positives"),
        ("4pos", "classwise_toparea_loio", None, "4 positives"),
        ("1pos_1neg", "classwise_toparea_loio", "realistic_hard", "1 positive + 1 realistic hard negative"),
        ("1pos_2neg", "classwise_toparea_loio", "realistic_hard", "1 positive + 2 realistic hard negatives"),
        ("1pos_3neg", "classwise_toparea_loio", "realistic_hard", "1 positive + 3 realistic hard negatives"),
        ("1pos_1strictneg", "classwise_toparea_loio", "realistic_hard_strict", "1 positive + 1 strict hard negative"),
        ("1pos_2strictneg", "classwise_toparea_loio", "realistic_hard_strict", "1 positive + 2 strict hard negatives"),
        ("1pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict", "1 positive + 3 strict hard negatives"),
        ("4pos_3neg", "classwise_toparea_loio", "realistic_hard", "4 positives + 3 realistic hard negatives"),
        ("1pos_1oracle_neg", "classwise_toparea_loio", "oracle_gt", "1 positive + 1 oracle negative"),
    ]
    for protocol, threshold, source, label in rows_to_show:
        row = _find_summary(summary, protocol, threshold, negative_source=source)
        lines.append(f"| {label} | {threshold} " + _metric_cells(row))

    lines.append("\n## Supervised Token Upper Bound\n\n")
    lines.append("| Variant | Folds | Segments | Accuracy | Macro F1 | Mean Dice | Macro mIoU |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    if supervised:
        row = supervised[0]
        lines.append(
            f"| {row.get('variant', '')} | {int(float(row.get('folds', 0)))} | {int(float(row.get('segments', 0)))} "
            f"| {_float(row, 'accuracy'):.4f} | {_float(row, 'macro_f1'):.4f} "
            f"| {_float(row, 'mean_dice_present_classes'):.4f} | {_float(row, 'macro_miou_present_classes'):.4f} |\n"
        )
    else:
        lines.append("| - | - | - | - | - | - | - |\n")

    lines.append("\n## Reference Baselines\n\n")
    lines.append("| Method | Threshold | mAP | Dice | mIoU | BF1@5 | BF1@10 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in aaai_summary:
        if (
            row.get("variant") in {"image_cell_reg_cellw0p5", "image_only", "cell_reg", "patch_only"}
            and row.get("prototype_protocol") == "median"
            and row.get("threshold_protocol") == "p90"
            and row.get("scope") == "exclude_prompt_region"
        ):
            lines.append(
                f"| {row.get('variant')} | P90 | {_float(row, 'mean_average_precision'):.4f} "
                f"| {_float(row, 'mean_dice'):.4f} | {_float(row, 'mean_binary_miou'):.4f} "
                f"| {_float(row, 'mean_boundary_f1_5px'):.4f} | {_float(row, 'mean_boundary_f1_10px'):.4f} |\n"
            )
    for name, rows in (("SAM ViT-B box", sam_summary), ("MedSAM ViT-B box", medsam_summary)):
        if rows:
            row = rows[0]
            lines.append(
                f"| {name} | local box | N/A | {_float(row, 'mean_dice'):.4f} "
                f"| {_float(row, 'mean_iou'):.4f} | {_float(row, 'mean_boundary_f1_5px'):.4f} "
                f"| {_float(row, 'mean_boundary_f1_10px'):.4f} |\n"
            )

    lines.append("\n## Per-Class Interaction Summary\n\n")
    lines.append("Per-class details are saved in `pret_aaai_next_by_class.csv`; use this table to identify which tissue classes benefit from extra positive or negative prompts.\n")
    if by_class:
        lines.append("\nThe report intentionally separates realistic interaction settings from oracle negative upper bounds.\n")

    report_path = output_dir / "PRET_SUPERPIXEL_AAAI_NEXT.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
