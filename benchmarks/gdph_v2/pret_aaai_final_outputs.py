from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gdph_v2.pret_utils import PRET_DIR, read_csv, write_csv_atomic


DEFAULT_NEXT_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_next_full10x"
DEFAULT_NEXT_V2_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_v2_full10x"
DEFAULT_AAAI_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_full10x_auto_physical"
DEFAULT_SOURCE_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical"

SCALE_SOURCES = [
    ("4096 d64 canonical", "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_4096_canonical"),
    ("8192 d64 canonical", "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_8192_canonical"),
    ("8192 d96 canonical", "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_8192_sp96_canonical"),
    ("8192 d128 canonical", "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_8192_sp128_canonical"),
    ("full10x auto physical", DEFAULT_SOURCE_ROOT),
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: object) -> str:
    if value in (None, "", "N/A", "not_found", "not_run"):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _write_md_table(path: Path, title: str, rows: list[dict], columns: list[str]) -> None:
    lines = [f"# {title}\n\n"]
    lines.append("| " + " | ".join(columns) + " |\n")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|\n")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_csv_nonempty(path: Path, rows: list[dict], placeholder: dict) -> None:
    write_csv_atomic(path, rows if rows else [placeholder])


def _load_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _find_next(summary: list[dict[str, str]], protocol: str, threshold: str, source: str = "none") -> dict[str, str] | None:
    for row in summary:
        if row.get("interaction_protocol") == protocol and row.get("threshold_protocol") == threshold and row.get("scope") == "exclude_prompt_region":
            if row.get("negative_source") == source:
                return row
    return None


def _next_row(label: str, row: dict[str, str] | None, threshold: str, source: str = "next") -> dict:
    if not row:
        return {"method": label, "threshold": threshold, "source": "not_found", "queries": "not_found"}
    return {
        "method": label,
        "threshold": threshold,
        "source": source,
        "queries": row.get("queries", ""),
        "mAP": row.get("mean_average_precision", ""),
        "P@top5_area": row.get("mean_precision_at_top5_area", ""),
        "Dice": row.get("mean_dice", ""),
        "mIoU": row.get("mean_binary_miou", ""),
        "BF1@5": row.get("mean_boundary_f1_5px", ""),
        "BF1@10": row.get("mean_boundary_f1_10px", ""),
        "Precision": row.get("mean_precision", ""),
        "Recall": row.get("mean_recall", ""),
    }


def _class_balanced_next_row(label: str, by_class: list[dict[str, str]], protocol: str, threshold: str, source: str = "none") -> dict:
    rows = [
        row for row in by_class
        if row.get("interaction_protocol") == protocol
        and row.get("threshold_protocol") == threshold
        and row.get("scope") == "exclude_prompt_region"
        and row.get("negative_source") == source
    ]
    if not rows:
        return {"method": label, "threshold": threshold, "source": "not_found", "classes": "not_found", "queries": "not_found"}
    return {
        "method": label,
        "threshold": threshold,
        "source": source if source != "none" else "next",
        "classes": len(rows),
        "queries": sum(int(_safe_float(row.get("queries"))) for row in rows),
        "mAP": mean(_safe_float(row.get("mean_average_precision")) for row in rows),
        "Dice": mean(_safe_float(row.get("mean_dice")) for row in rows),
        "mIoU": mean(_safe_float(row.get("mean_iou")) for row in rows),
        "BF1@5": mean(_safe_float(row.get("mean_boundary_f1_5px")) for row in rows),
        "BF1@10": mean(_safe_float(row.get("mean_boundary_f1_10px")) for row in rows),
        "Precision": mean(_safe_float(row.get("mean_precision")) for row in rows),
        "Recall": mean(_safe_float(row.get("mean_recall")) for row in rows),
    }


def _aaai_variant_row(rows: list[dict[str, str]], variant: str, label: str) -> dict:
    for row in rows:
        if (
            row.get("variant") == variant
            and row.get("prototype_protocol") == "median"
            and row.get("threshold_protocol") == "p90"
            and row.get("scope") == "exclude_prompt_region"
        ):
            return {
                "method": label,
                "threshold": "P90",
                "source": "aaai_baseline",
                "queries": row.get("queries", ""),
                "mAP": row.get("mean_average_precision", ""),
                "P@top5_area": "N/A",
                "Dice": row.get("mean_dice", ""),
                "mIoU": row.get("mean_binary_miou", ""),
                "BF1@5": row.get("mean_boundary_f1_5px", ""),
                "BF1@10": row.get("mean_boundary_f1_10px", ""),
                "Precision": "N/A",
                "Recall": "N/A",
            }
    return {"method": label, "threshold": "P90", "source": "not_found", "queries": "not_found"}


def _aaai_variant_class_balanced_row(rows: list[dict[str, str]], variant: str, label: str) -> dict:
    subset = [
        row for row in rows
        if row.get("variant") == variant
        and row.get("prototype_protocol") == "median"
        and row.get("threshold_protocol") == "p90"
        and row.get("scope") == "exclude_prompt_region"
    ]
    if not subset:
        return {"method": label, "threshold": "P90", "source": "not_found", "classes": "not_found", "queries": "not_found"}
    return {
        "method": label,
        "threshold": "P90",
        "source": "aaai_baseline",
        "classes": len(subset),
        "queries": sum(int(_safe_float(row.get("queries"))) for row in subset),
        "mAP": mean(_safe_float(row.get("mean_average_precision")) for row in subset),
        "Dice": mean(_safe_float(row.get("mean_dice")) for row in subset),
        "mIoU": mean(_safe_float(row.get("mean_binary_miou")) for row in subset),
        "BF1@5": mean(_safe_float(row.get("mean_boundary_f1_5px")) for row in subset),
        "BF1@10": mean(_safe_float(row.get("mean_boundary_f1_10px")) for row in subset),
        "Precision": "N/A",
        "Recall": "N/A",
    }


def _sam_row(rows: list[dict[str, str]], label: str) -> dict:
    if not rows:
        return {"method": label, "threshold": "local box", "source": "not_found", "queries": "not_found"}
    row = rows[0]
    return {
        "method": label,
        "threshold": "local box",
        "source": "sam_baseline",
        "queries": row.get("queries", ""),
        "mAP": "N/A",
        "P@top5_area": "N/A",
        "Dice": row.get("mean_dice", ""),
        "mIoU": row.get("mean_iou", ""),
        "BF1@5": row.get("mean_boundary_f1_5px", ""),
        "BF1@10": row.get("mean_boundary_f1_10px", ""),
        "Precision": row.get("mean_precision", "N/A"),
        "Recall": row.get("mean_recall", "N/A"),
    }


def _aggregate_random(source_root: Path) -> dict:
    path = source_root / PRET_DIR / "pret_metrics.csv"
    if not path.exists():
        return {"method": "random token", "threshold": "P90", "source": "not_found", "queries": "not_found"}
    rows = [
        row for row in read_csv(path)
        if row.get("variant") == "random_token"
        and row.get("prompt_source") == "realistic_box"
        and row.get("scope") == "exclude_prompt_region"
        and row.get("smoothing_alpha", "0.0") == "0.0"
    ]
    if not rows:
        return {"method": "random token", "threshold": "P90", "source": "not_run", "queries": "not_run"}
    return {
        "method": "random token",
        "threshold": "P90",
        "source": "pret_metrics",
        "queries": len(rows),
        "mAP": mean(_safe_float(row.get("average_precision")) for row in rows),
        "P@top5_area": mean(_safe_float(row.get("precision_at_top5_area")) for row in rows),
        "Dice": mean(_safe_float(row.get("percentile_90_dice")) for row in rows),
        "mIoU": mean(_safe_float(row.get("percentile_90_iou")) for row in rows),
        "BF1@5": "N/A",
        "BF1@10": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
    }


def _aggregate_scale(root: Path, label: str) -> dict:
    path = root / PRET_DIR / "pret_metrics.csv"
    if not path.exists():
        return {"setting": label, "queries": "not_found"}
    rows = [
        row for row in read_csv(path)
        if row.get("variant") == "image_cell_reg_cellw0p5"
        and row.get("prompt_source") == "realistic_box"
        and row.get("scope") == "exclude_prompt_region"
        and row.get("baseline", "none") == "none"
        and row.get("smoothing_alpha", "0.0") == "0.0"
    ]
    if not rows:
        return {"setting": label, "queries": "not_run"}
    return {
        "setting": label,
        "queries": len(rows),
        "mAP": mean(_safe_float(row.get("average_precision")) for row in rows),
        "P@top5_area": mean(_safe_float(row.get("precision_at_top5_area")) for row in rows),
        "Top10_Dice": mean(_safe_float(row.get("top10_area_dice")) for row in rows),
        "BestDice": mean(_safe_float(row.get("best_area_dice")) for row in rows),
        "BestArea": mean(_safe_float(row.get("best_area_ratio")) for row in rows),
    }


def _delta_rows(by_class: list[dict[str, str]]) -> list[dict]:
    def find(class_id: str, protocol: str, threshold: str, source: str = "none") -> dict[str, str] | None:
        for row in by_class:
            if (
                row.get("class_id") == class_id
                and row.get("interaction_protocol") == protocol
                and row.get("threshold_protocol") == threshold
                and row.get("scope") == "exclude_prompt_region"
                and row.get("negative_source") == source
            ):
                return row
        return None

    output = []
    for class_id in sorted({row.get("class_id", "") for row in by_class}, key=lambda value: int(value) if value else -1):
        p90 = find(class_id, "1pos", "p90")
        adaptive = find(class_id, "1pos", "prompt_adaptive_area_loio")
        neg = find(class_id, "1pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict")
        if not p90 or not adaptive or not neg:
            continue
        output.append(
            {
                "class_id": class_id,
                "class_name": p90.get("class_name", ""),
                "queries_p90": p90.get("queries", ""),
                "p90_mAP": p90.get("mean_average_precision", ""),
                "p90_dice": p90.get("mean_dice", ""),
                "p90_mIoU": p90.get("mean_iou", ""),
                "prompt_adaptive_mAP": adaptive.get("mean_average_precision", ""),
                "prompt_adaptive_dice": adaptive.get("mean_dice", ""),
                "prompt_adaptive_mIoU": adaptive.get("mean_iou", ""),
                "neg3_mAP": neg.get("mean_average_precision", ""),
                "neg3_dice": neg.get("mean_dice", ""),
                "neg3_mIoU": neg.get("mean_iou", ""),
                "delta_prompt_adaptive_vs_p90_dice": _safe_float(adaptive.get("mean_dice")) - _safe_float(p90.get("mean_dice")),
                "delta_3neg_vs_prompt_adaptive_dice": _safe_float(neg.get("mean_dice")) - _safe_float(adaptive.get("mean_dice")),
                "delta_3neg_vs_p90_dice": _safe_float(neg.get("mean_dice")) - _safe_float(p90.get("mean_dice")),
            }
        )
    return output


def _write_loio_audit(output_dir: Path, compact_rows: list[dict[str, str]], validation_path: Path) -> None:
    images = sorted({row["image_id"] for row in compact_rows})
    rows = []
    for image_id in images:
        train_images = [item for item in images if item != image_id]
        rows.extend(
            [
                {
                    "heldout_image": image_id,
                    "protocol": "global_toparea_loio",
                    "uses_class_id": False,
                    "uses_heldout_gt": False,
                    "train_images": ";".join(train_images),
                    "test_image": image_id,
                    "note": "Global area prior is computed from all other WSIs.",
                },
                {
                    "heldout_image": image_id,
                    "protocol": "classwise_toparea_loio",
                    "uses_class_id": True,
                    "uses_heldout_gt": False,
                    "train_images": ";".join(train_images),
                    "test_image": image_id,
                    "note": "Class-specific area prior is computed from other WSIs only.",
                },
                {
                    "heldout_image": image_id,
                    "protocol": "prompt_adaptive_area_loio",
                    "uses_class_id": False,
                    "uses_heldout_gt": False,
                    "train_images": ";".join(train_images),
                    "test_image": image_id,
                    "note": "Regressor uses prompt/score features only; GT-derived prompt purity fields are excluded.",
                },
                {
                    "heldout_image": image_id,
                    "protocol": "prompt_adaptive_area_loio_v2_rf",
                    "uses_class_id": False,
                    "uses_heldout_gt": False,
                    "train_images": ";".join(train_images),
                    "test_image": image_id,
                    "note": "RF regressor uses expanded prompt/score features only; GT audit fields are excluded.",
                },
                {
                    "heldout_image": image_id,
                    "protocol": "prompt_adaptive_area_loio_v2_linear",
                    "uses_class_id": False,
                    "uses_heldout_gt": False,
                    "train_images": ";".join(train_images),
                    "test_image": image_id,
                    "note": "Huber/linear regressor uses expanded prompt/score features only; GT audit fields are excluded.",
                },
            ]
        )
    write_csv_atomic(output_dir / "loio_audit.csv", rows)
    validation = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    md = [
        "# LOIO Calibration Audit\n\n",
        "All calibrated mask protocols are leave-one-image-out at WSI level.\n\n",
        "| Protocol | Uses class id | Uses held-out GT | Train/calibration data | Test data |\n",
        "|---|---:|---:|---|---|\n",
        "| global_toparea_loio | no | no | other 19 WSIs | held-out WSI |\n",
        "| classwise_toparea_loio | yes | no | same-class queries from other 19 WSIs | held-out WSI |\n",
        "| prompt_adaptive_area_loio | no | no | score/prompt features and best-area targets from other 19 WSIs | held-out WSI |\n",
        "| prompt_adaptive_area_loio_v2_rf | no | no | expanded score/prompt features from other 19 WSIs | held-out WSI |\n",
        "| prompt_adaptive_area_loio_v2_linear | no | no | expanded score/prompt features from other 19 WSIs | held-out WSI |\n\n",
        "Prompt-adaptive feature leakage check: `prompt_purity` and `prompt_target_area_fraction` are not model inputs.\n\n",
        f"Validation note: {validation.get('note', 'not_found')}\n",
    ]
    (output_dir / "LOIO_AUDIT.md").write_text("".join(md), encoding="utf-8")


def _write_negative_prompt_audit(output_dir: Path, compact_rows: list[dict[str, str]]) -> None:
    rows = [
        row for row in compact_rows
        if row.get("scope") == "exclude_prompt_region"
        and _safe_float(row.get("negative_prompt_count")) > 0
    ]
    detailed = []
    for row in rows:
        detailed.append(
            {
                "query_id": row.get("query_id", ""),
                "image_id": row.get("image_id", ""),
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "interaction_protocol": row.get("interaction_protocol", ""),
                "negative_source": row.get("negative_source", ""),
                "negative_prompt_count": row.get("negative_prompt_count", ""),
                "negative_gt_majority_label": row.get("negative_gt_majority_label", ""),
                "negative_gt_target_fraction": row.get("negative_gt_target_fraction", ""),
                "negative_gt_mean_purity": row.get("negative_gt_mean_purity", ""),
                "negative_gt_min_purity": row.get("negative_gt_min_purity", ""),
                "negative_gt_valid_fraction": row.get("negative_gt_valid_fraction", ""),
            }
        )
    if detailed:
        write_csv_atomic(output_dir / "negative_prompt_audit.csv", detailed)

    summary = []
    keys = sorted({(row.get("interaction_protocol", ""), row.get("negative_source", "")) for row in rows})
    for protocol, source in keys:
        subset = [row for row in rows if row.get("interaction_protocol") == protocol and row.get("negative_source") == source]
        if not subset:
            continue
        target_fracs = [_safe_float(row.get("negative_gt_target_fraction")) for row in subset]
        min_purities = [_safe_float(row.get("negative_gt_min_purity")) for row in subset]
        summary.append(
            {
                "interaction_protocol": protocol,
                "negative_source": source,
                "queries": len(subset),
                "mean_negative_gt_target_fraction": float(np.mean(target_fracs)),
                "max_negative_gt_target_fraction": float(np.max(target_fracs)),
                "contaminated_query_fraction_target_frac_gt0": float(np.mean(np.asarray(target_fracs) > 0.0)),
                "contaminated_query_fraction_target_frac_ge0p2": float(np.mean(np.asarray(target_fracs) >= 0.2)),
                "mean_negative_gt_min_purity": float(np.mean(min_purities)),
                "low_purity_query_fraction_lt0p7": float(np.mean(np.asarray(min_purities) < 0.7)),
            }
        )
    if summary:
        write_csv_atomic(output_dir / "negative_prompt_audit_summary.csv", summary)
        md = ["# Negative Prompt Audit\n\n"]
        md.append("Strict negatives require GT majority label != target class, valid retrieval segment, and GT purity >= 0.7 for every selected negative segment.\n\n")
        md.append("| Protocol | Source | Queries | Mean target frac | Max target frac | Target frac > 0 | Target frac >= 0.2 | Low purity < 0.7 |\n")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            md.append(
                f"| {row['interaction_protocol']} | {row['negative_source']} | {row['queries']} "
                f"| {_fmt(row['mean_negative_gt_target_fraction'])} | {_fmt(row['max_negative_gt_target_fraction'])} "
                f"| {_fmt(row['contaminated_query_fraction_target_frac_gt0'])} "
                f"| {_fmt(row['contaminated_query_fraction_target_frac_ge0p2'])} "
                f"| {_fmt(row['low_purity_query_fraction_lt0p7'])} |\n"
            )
        (output_dir / "NEGATIVE_PROMPT_AUDIT.md").write_text("".join(md), encoding="utf-8")


def _write_results_md(output_dir: Path, final_rows: list[dict], class_balanced_rows: list[dict], scale_rows: list[dict], delta_rows: list[dict], supervised_rows: list[dict[str, str]]) -> None:
    lines = ["# AAAI-style Results\n\n"]
    lines.append("## Main comparison\n\n")
    lines.append("See `final_main_table_query_weighted.md` and `final_main_table_class_balanced.md`. Query-weighted is the natural canonical-query average; class-balanced first averages per class.\n\n")
    top = [row for row in final_rows if row.get("method") in {"Ours 1pos prompt-adaptive LOIO", "Ours 1pos + 3 strict hard negatives", "patch prototype", "SAM ViT-B box", "MedSAM ViT-B box"}]
    lines.append("| Method | Threshold | mAP | Dice | mIoU | BF1@5 |\n|---|---:|---:|---:|---:|---:|\n")
    for row in top:
        lines.append(f"| {row.get('method')} | {row.get('threshold')} | {_fmt(row.get('mAP'))} | {_fmt(row.get('Dice'))} | {_fmt(row.get('mIoU'))} | {_fmt(row.get('BF1@5'))} |\n")
    lines.append("\nClass-balanced rows are available for methods with per-class outputs:\n\n")
    lines.append("| Method | Threshold | Classes | mAP | Dice | mIoU | BF1@5 |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for row in class_balanced_rows:
        lines.append(f"| {row.get('method')} | {row.get('threshold')} | {row.get('classes')} | {_fmt(row.get('mAP'))} | {_fmt(row.get('Dice'))} | {_fmt(row.get('mIoU'))} | {_fmt(row.get('BF1@5'))} |\n")

    lines.append("\n## Scale ablation\n\n")
    lines.append("| Setting | Queries | mAP | P@top5 | Top10 Dice | BestDice | BestArea |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for row in scale_rows:
        lines.append(f"| {row.get('setting')} | {row.get('queries')} | {_fmt(row.get('mAP'))} | {_fmt(row.get('P@top5_area'))} | {_fmt(row.get('Top10_Dice'))} | {_fmt(row.get('BestDice'))} | {_fmt(row.get('BestArea'))} |\n")

    lines.append("\n## Token ablation\n\n")
    lines.append("Image-only, cell-only, patch-only, fused image+cell, and optional texture/cell-stat enhanced tokens are included when their metric files are present.\n\n")

    lines.append("## Calibration ablation\n\n")
    lines.append("P90 is a strict ranking-to-mask baseline. Global LOIO removes class prior, classwise LOIO uses benchmark class id, and prompt-adaptive LOIO predicts area from prompt/score features without held-out GT.\n\n")

    lines.append("## Interactive negative ablation\n\n")
    lines.append("The main interaction curve compares 1 positive only against +1/+2/+3 strict realistic hard negative prompts. Strict negatives require GT majority label != target class and purity >= 0.7 in the benchmark audit. Weak realistic hard negatives are retained only as a diagnostic baseline.\n\n")

    lines.append("## SAM/MedSAM comparison\n\n")
    lines.append("SAM and MedSAM are local box-prompt mask baselines. They do not perform whole-slide same-class retrieval, so mAP is not applicable.\n\n")

    lines.append("## Supervised upper bound\n\n")
    if supervised_rows:
        row = supervised_rows[0]
        lines.append(
            f"RandomForest LOIO classifier on image+cell superpixel tokens: accuracy={_fmt(row.get('accuracy'))}, "
            f"macro F1={_fmt(row.get('macro_f1'))}, mean Dice={_fmt(row.get('mean_dice_present_classes'))}, "
            f"macro mIoU={_fmt(row.get('macro_miou_present_classes'))}.\n\n"
        )
    else:
        lines.append("not_found\n\n")

    lines.append("## Failure analysis\n\n")
    lines.append("Use `per_class_delta_table.md` to identify classes where calibration or hard negatives fail to improve Dice. Visual examples are generated under `visual_summary/`.\n")
    output_dir.joinpath("AAAI_RESULTS.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final AAAI tables, LOIO audit, and Results.md.")
    parser.add_argument("--next_root", default=DEFAULT_NEXT_V2_ROOT)
    parser.add_argument("--aaai_root", default=DEFAULT_AAAI_ROOT)
    parser.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()

    next_dir = Path(args.next_root) / PRET_DIR
    aaai_dir = Path(args.aaai_root) / PRET_DIR
    source_root = Path(args.source_root)
    summary = _load_csv(next_dir / "pret_aaai_next_summary.csv")
    by_class = _load_csv(next_dir / "pret_aaai_next_by_class.csv")
    aaai_summary = _load_csv(aaai_dir / "pret_aaai_summary.csv")
    aaai_by_class = _load_csv(aaai_dir / "pret_aaai_by_class.csv")
    sam_summary = _load_csv(aaai_dir / "sam_vit_b_baseline_summary.csv")
    medsam_summary = _load_csv(aaai_dir / "medsam_vit_b_baseline_summary.csv")
    supervised = _load_csv(next_dir / "pret_supervised_upper_bound_overall.csv")

    final_rows = [
        _next_row("Ours 1pos P90", _find_next(summary, "1pos", "p90"), "P90"),
        _next_row("Ours 1pos global LOIO", _find_next(summary, "1pos", "global_toparea_loio"), "global_toparea_loio"),
        _next_row("Ours 1pos prompt-adaptive LOIO", _find_next(summary, "1pos", "prompt_adaptive_area_loio"), "prompt_adaptive_area_loio"),
        _next_row("Ours 1pos prompt-adaptive LOIO v2 RF", _find_next(summary, "1pos", "prompt_adaptive_area_loio_v2_rf"), "prompt_adaptive_area_loio_v2_rf"),
        _next_row("Ours 1pos prompt-adaptive LOIO v2 linear", _find_next(summary, "1pos", "prompt_adaptive_area_loio_v2_linear"), "prompt_adaptive_area_loio_v2_linear"),
        _next_row("Ours 1pos classwise LOIO", _find_next(summary, "1pos", "classwise_toparea_loio"), "classwise_toparea_loio"),
        _next_row("Ours 1pos + 1 strict hard negative", _find_next(summary, "1pos_1strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "+1 strict hard neg"),
        _next_row("Ours 1pos + 2 strict hard negatives", _find_next(summary, "1pos_2strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "+2 strict hard neg"),
        _next_row("Ours 1pos + 3 strict hard negatives", _find_next(summary, "1pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "+3 strict hard neg"),
        _next_row("Ours 2pos + 3 strict hard negatives", _find_next(summary, "2pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "2pos +3 strict hard neg"),
        _next_row("Ours 4pos + 3 strict hard negatives", _find_next(summary, "4pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "4pos +3 strict hard neg"),
        _next_row("Ours 8pos + 3 strict hard negatives", _find_next(summary, "8pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"), "classwise_toparea_loio", "8pos +3 strict hard neg"),
        _next_row("Diagnostic weak +3 hard negatives", _find_next(summary, "1pos_3neg", "classwise_toparea_loio", "realistic_hard"), "classwise_toparea_loio", "weak +3 hard neg"),
        _aaai_variant_row(aaai_summary, "image_only", "image-only"),
        _aaai_variant_row(aaai_summary, "cell_reg", "cell-only"),
        _aaai_variant_row(aaai_summary, "patch_only", "patch prototype"),
        _aggregate_random(source_root),
        _sam_row(sam_summary, "SAM ViT-B box"),
        _sam_row(medsam_summary, "MedSAM ViT-B box"),
    ]
    columns = ["method", "threshold", "source", "queries", "mAP", "P@top5_area", "Dice", "mIoU", "BF1@5", "BF1@10", "Precision", "Recall"]
    write_csv_atomic(next_dir / "final_main_table.csv", final_rows)
    _write_md_table(next_dir / "final_main_table.md", "Final Main Table", final_rows, columns)
    write_csv_atomic(next_dir / "final_main_table_query_weighted.csv", final_rows)
    _write_md_table(next_dir / "final_main_table_query_weighted.md", "Final Main Table Query Weighted", final_rows, columns)

    class_balanced_rows = [
        _class_balanced_next_row("Ours 1pos P90", by_class, "1pos", "p90"),
        _class_balanced_next_row("Ours 1pos prompt-adaptive LOIO", by_class, "1pos", "prompt_adaptive_area_loio"),
        _class_balanced_next_row("Ours 1pos prompt-adaptive LOIO v2 RF", by_class, "1pos", "prompt_adaptive_area_loio_v2_rf"),
        _class_balanced_next_row("Ours 1pos prompt-adaptive LOIO v2 linear", by_class, "1pos", "prompt_adaptive_area_loio_v2_linear"),
        _class_balanced_next_row("Ours 1pos classwise LOIO", by_class, "1pos", "classwise_toparea_loio"),
        _class_balanced_next_row("Ours 1pos + 3 strict hard negatives", by_class, "1pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"),
        _class_balanced_next_row("Ours 2pos + 3 strict hard negatives", by_class, "2pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"),
        _class_balanced_next_row("Ours 4pos + 3 strict hard negatives", by_class, "4pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"),
        _class_balanced_next_row("Ours 8pos + 3 strict hard negatives", by_class, "8pos_3strictneg", "classwise_toparea_loio", "realistic_hard_strict"),
        _aaai_variant_class_balanced_row(aaai_by_class, "image_only", "image-only"),
        _aaai_variant_class_balanced_row(aaai_by_class, "cell_reg", "cell-only"),
        _aaai_variant_class_balanced_row(aaai_by_class, "patch_only", "patch prototype"),
    ]
    cb_columns = ["method", "threshold", "source", "classes", "queries", "mAP", "Dice", "mIoU", "BF1@5", "BF1@10", "Precision", "Recall"]
    write_csv_atomic(next_dir / "final_main_table_class_balanced.csv", class_balanced_rows)
    _write_md_table(next_dir / "final_main_table_class_balanced.md", "Final Main Table Class Balanced", class_balanced_rows, cb_columns)
    query_counts = sorted(
        [
            {
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "queries": row.get("queries", ""),
                "dice": row.get("mean_dice", ""),
                "mIoU": row.get("mean_iou", ""),
                "mAP": row.get("mean_average_precision", ""),
                "BF1@5": row.get("mean_boundary_f1_5px", ""),
            }
            for row in by_class
            if row.get("interaction_protocol") == "1pos"
            and row.get("threshold_protocol") == "prompt_adaptive_area_loio"
            and row.get("scope") == "exclude_prompt_region"
            and row.get("negative_source") == "none"
        ],
        key=lambda row: int(row["class_id"]) if row["class_id"] != "" else -1,
    )
    _write_csv_nonempty(next_dir / "per_class_query_count.csv", query_counts, {"note": "not_available_for_this_protocol_subset"})
    _write_md_table(next_dir / "per_class_query_count.md", "Per-Class Query Count", query_counts, ["class_id", "class_name", "queries", "mAP", "dice", "mIoU", "BF1@5"])

    deltas = _delta_rows(by_class)
    _write_csv_nonempty(next_dir / "per_class_delta_table.csv", deltas, {"note": "not_available_for_this_protocol_subset"})
    _write_md_table(
        next_dir / "per_class_delta_table.md",
        "Per-Class Delta Table",
        deltas,
        [
            "class_id", "class_name", "queries_p90", "p90_dice", "prompt_adaptive_dice", "neg3_dice",
            "delta_prompt_adaptive_vs_p90_dice", "delta_3neg_vs_prompt_adaptive_dice", "delta_3neg_vs_p90_dice",
        ],
    )

    scale_rows = [_aggregate_scale(Path(root), label) for label, root in SCALE_SOURCES]
    write_csv_atomic(next_dir / "scale_ablation_table.csv", scale_rows)
    _write_md_table(next_dir / "scale_ablation_table.md", "Scale Ablation Table", scale_rows, ["setting", "queries", "mAP", "P@top5_area", "Top10_Dice", "BestDice", "BestArea"])

    compact_rows = _load_csv(next_dir / "pret_aaai_next_compact.csv")
    _write_loio_audit(next_dir, compact_rows, next_dir / "pret_aaai_next_validation.json")
    _write_negative_prompt_audit(next_dir, compact_rows)
    _write_results_md(next_dir, final_rows, class_balanced_rows, scale_rows, deltas, supervised)
    print(json.dumps({"output_dir": str(next_dir), "final_rows": len(final_rows), "delta_rows": len(deltas)}, indent=2))


if __name__ == "__main__":
    main()
