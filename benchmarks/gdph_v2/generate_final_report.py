from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _runtime_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "logs").glob("*.status.json")):
        payload = _load(path)
        if (
            payload.get("state") != "completed"
            or not payload.get("started_at")
            or not payload.get("finished_at")
        ):
            continue
        started = datetime.fromisoformat(payload["started_at"])
        finished = datetime.fromisoformat(payload["finished_at"])
        rows.append(
            {
                "stage": payload.get("stage", path.name.removesuffix(".status.json")),
                "started_at": payload["started_at"],
                "finished_at": payload["finished_at"],
                "elapsed_seconds": (finished - started).total_seconds(),
                "status_file": str(path),
            }
        )
    return rows


def generate_report(root: Path) -> str:
    setup = _load(root / "config" / "experiment.json")
    scale = _load(root / "cell_classification" / "scale_comparison" / "summary.json")
    classification = {
        head: _load(root / "cell_classification" / f"{head}_balanced_valid_only" / "metrics.json")
        for head in ("raw", "reg", "proj")
    }
    retrieval = _load(root / "region_retrieval" / "cell_retrieval_summary.json")
    coverage = _load(root / "dense_evaluation" / "cell_coverage_summary.json")
    coverage_by_class = _load(root / "dense_evaluation" / "cell_coverage_by_class.json")
    hybrid = _load(root / "dense_evaluation" / "patch_hybrid_summary.json")
    hybrid_by_class = _load(root / "dense_evaluation" / "patch_hybrid_by_class.json")
    palette = _load(root / "manifests" / "gdph_tissue_palette.json")
    class_names = {int(item["id"]): item["name"] for item in palette["classes"]}
    runtimes = _runtime_rows(root)
    lines = [
        "# CellAtlas GDPH External Validation v2",
        "",
        "## Experiment identity",
        "",
        f"- Output root: `{root}`",
        f"- XCellFormer SHA256: `{setup['models']['xcellformer']['sha256']}`",
        f"- CTransPath SHA256: `{setup['models']['ctranspath']['sha256']}`",
        f"- Cellpose SHA256: `{setup['models']['cellpose']['sha256']}`",
        "- GDPH is external to XCellFormer training.",
        "- Inference uses original level-0 HE; coordinates are mapped to 10x GT after inference.",
        "",
        "## Runtime and outputs",
        "",
        f"- Experiment output root: `{root}`",
        f"- Machine-readable audit: `{root / 'config' / 'experiment_audit.json'}`",
        f"- Per-cell outputs: `{root / 'cells'}`",
        f"- Metrics: `{root / 'cell_classification'}`, `{root / 'region_retrieval'}`, `{root / 'dense_evaluation'}`",
        "",
        "| Stage | Started | Finished | Wall time |",
        "|---|---|---|---:|",
    ]
    for runtime in runtimes:
        lines.append(
            f"| {runtime['stage']} | {runtime['started_at']} | {runtime['finished_at']} | "
            f"{runtime['elapsed_seconds'] / 60:.2f} min |"
        )
    lines.extend([
        "",
        "## Original-resolution vs resized nucleus detection",
        "",
        "| Distance | Method | Precision | Recall | F1 |",
        "|---:|---|---:|---:|---:|",
    ])
    for threshold, methods in scale.items():
        for method in ("fullres", "resized_10x"):
            values = methods[method]
            lines.append(
                f"| {threshold} px | {method} | {_percent(values['mean_precision'])} | "
                f"{_percent(values['mean_recall'])} | {_percent(values['mean_f1'])} |"
            )
        lines.append(
            f"| {threshold} px | paired ΔF1 | — | — | "
            f"{_percent(methods['paired_delta_mean_f1_fullres_minus_resized'])} |"
        )
    lines.extend(
        [
            "",
            "## Five-fold cell-level tissue classification",
            "",
            "| Head | Pooled accuracy | Pooled mIoU | Pooled macro-F1 | Image mIoU | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for head, values in classification.items():
        ci = values["image_mean_iou_ci95"]
        lines.append(
            f"| {head} | {_percent(values['pooled_accuracy'])} | {_percent(values['pooled_mean_iou'])} | "
            f"{_percent(values['pooled_macro_f1'])} | {_percent(values['image_mean_iou'])} | "
            f"{_percent(ci[0])}–{_percent(ci[1])} |"
        )
    lines.extend(
        [
            "",
            "## Query-by-region cell retrieval",
            "",
            "| Head | Queries | mAP | AUROC | Binary accuracy | Binary F1 | Binary IoU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for head, values in retrieval.items():
        lines.append(
            f"| {head} | {values['queries']} | {_percent(values['mean_average_precision'])} | "
            f"{_percent(values['mean_auroc'])} | {_percent(values['mean_binary_accuracy'])} | "
            f"{_percent(values['mean_binary_f1'])} | {_percent(values['mean_binary_iou'])} |"
        )
    lines.extend(
        [
            "",
            "## Cell coverage and oracle propagation ceiling",
            "",
            "| Radius at 10x | Coverage | Covered oracle accuracy | Whole-image oracle accuracy | Whole-image oracle mIoU |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for values in coverage:
        lines.append(
            f"| {values['radius_gt_pixels']} px | {_percent(values['mean_coverage'])} | "
            f"{_percent(values['mean_covered_oracle_accuracy'])} | "
            f"{_percent(values['mean_whole_image_oracle_accuracy'])} | "
            f"{_percent(values['mean_whole_image_oracle_miou'])} |"
        )
    lines.extend(
        [
            "",
            "### Low-cell/acellular tissue coverage",
            "",
            "| Radius at 10x | Class | Coverage | Whole-image oracle accuracy | Mean nearest-cell p90 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for values in coverage_by_class:
        if values["acellular_focus"]:
            lines.append(
                f"| {values['radius_gt_pixels']} px | {values['class_name']} | "
                f"{_percent(values['mean_coverage'])} | "
                f"{_percent(values['mean_whole_image_oracle_accuracy'])} | "
                f"{values['mean_nearest_cell_distance_p90']:.1f} px |"
            )
    lines.extend(
        [
            "",
            "## Patch and cell+patch retrieval",
            "",
            "| Method | Queries | mAP | AUROC | Binary F1 | Binary IoU | Cell coverage at patches |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in hybrid.items():
        lines.append(
            f"| {method} | {values['queries']} | {_percent(values['mean_average_precision'])} | "
            f"{_percent(values['mean_auroc'])} | {_percent(values['mean_binary_f1'])} | "
            f"{_percent(values['mean_binary_iou'])} | {_percent(values['mean_cell_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "### Patch/hybrid retrieval in low-cell tissues",
            "",
            "| Method | Class | Queries | mAP | Binary F1 | Binary IoU | Cell coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for values in hybrid_by_class:
        if values["acellular_focus"]:
            lines.append(
                f"| {values['method']} | {class_names[int(values['class_id'])]} | "
                f"{values['queries']} | {_percent(values['mean_average_precision'])} | "
                f"{_percent(values['mean_binary_f1'])} | "
                f"{_percent(values['mean_binary_iou'])} | "
                f"{_percent(values['mean_cell_coverage'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Cell-level metrics describe tissue classification at detected nuclei, not dense tissue segmentation.",
            "- Coverage and oracle metrics quantify where cell-only propagation is structurally incapable of predicting.",
            "- Patch and hybrid results cover low-cell regions such as fat, mucus, and necrotic cavities.",
            "- Query regions are excluded from retrieval candidates to avoid trivial self-retrieval.",
            "- Binary retrieval thresholds use only the query-region similarity distribution (10th percentile), never candidate GT.",
            "- Automatic query boxes are GT-screened simulations of user selections; GT is not used in retrieval scoring.",
            "- Patch inference skips only near-pure-white patches (white ratio >= 99.5%) to retain fat and other low-cell tissue.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final GDPH v2 report from verified metrics.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    root = Path(args.output_root)
    report = generate_report(root)
    runtimes = _runtime_rows(root)
    runtime_path = root / "config" / "runtime_summary.json"
    temporary_runtime = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
    temporary_runtime.write_text(
        json.dumps(runtimes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_runtime.replace(runtime_path)
    output_path = root / "FINAL_REPORT.md"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        file.write(report)
    temporary.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
