#!/usr/bin/env python3
"""Aggregate completed TLS retrieval reports without changing case artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


COUNT_METRICS = (
    "union",
    "prompted_union",
    "held_out_union",
    "held_out_censored_prompted_region",
)
SCALAR_METRICS = (
    "prompted_macro_recall",
    "held_out_macro_recall",
    "held_out_retrieved_recall_ge_0_1",
    "held_out_retrieved_recall_ge_0_5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=30)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def safe_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("distribution input is empty or non-finite")
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    output = args.output_root / f"tls30_metrics_{args.timestamp}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)

    report_paths = [path.resolve() for path in args.reports]
    if len(report_paths) != args.expected_cases or len(set(report_paths)) != len(report_paths):
        raise ValueError(
            f"expected {args.expected_cases} unique reports, got {len(report_paths)}"
        )
    case_rows: list[dict] = []
    polygon_rows: list[dict] = []
    identities: set[str] = set()
    for report_path in sorted(report_paths):
        report = json.loads(report_path.read_text())
        case_manifest_path = Path(report["case_manifest"])
        case = json.loads(case_manifest_path.read_text())
        identity = str(Path(case["wsi_path"]).resolve())
        if identity in identities:
            raise ValueError(f"duplicate WSI report: {identity}")
        identities.add(identity)
        wsi_id = Path(case["wsi_path"]).stem.split()[0]
        row: dict[str, object] = {
            "wsi_id": wsi_id,
            "wsi_path": case["wsi_path"],
            "report_path": str(report_path),
            "case_manifest": str(case_manifest_path),
            "annotation_count": int(case["annotation_count"]),
            "prompted_polygon_count": int(report["prompted_polygon_count"]),
            "held_out_polygon_count": int(report["held_out_polygon_count"]),
        }
        for metric in COUNT_METRICS:
            value = report.get(metric)
            if value is None:
                raise ValueError(f"{metric} missing for {wsi_id}")
            for field in ("tp", "fp", "fn", "dice", "recall", "precision"):
                row[f"{metric}_{field}"] = value[field]
        for metric in SCALAR_METRICS:
            value = report.get(metric)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"{metric} missing/non-finite for {wsi_id}")
            row[metric] = float(value)
        case_rows.append(row)
        for polygon in report["per_polygon"]:
            polygon_rows.append(
                {
                    "wsi_id": wsi_id,
                    "polygon_index": int(polygon["polygon_index"]),
                    "prompted": bool(polygon["prompted"]),
                    "area_pixels_10x": int(polygon["area_pixels_10x"]),
                    "tp": int(polygon["tp"]),
                    "fp": int(polygon["fp"]),
                    "fn": int(polygon["fn"]),
                    "dice": float(polygon["dice"]),
                    "recall": float(polygon["recall"]),
                    "precision": float(polygon["precision"]),
                }
            )

    cases = pd.DataFrame(case_rows).sort_values("wsi_id").reset_index(drop=True)
    polygons = pd.DataFrame(polygon_rows).sort_values(
        ["wsi_id", "polygon_index"]
    ).reset_index(drop=True)
    if int(cases["annotation_count"].sum()) != len(polygons):
        raise ValueError("case annotation counts do not match per-polygon rows")
    if not (
        cases["prompted_polygon_count"] + cases["held_out_polygon_count"]
        == cases["annotation_count"]
    ).all():
        raise ValueError("prompted/held-out polygon partition is incomplete")

    macro: dict[str, dict] = {}
    micro: dict[str, dict] = {}
    for metric in COUNT_METRICS:
        for field in ("dice", "recall", "precision"):
            macro[f"{metric}_{field}"] = distribution(
                cases[f"{metric}_{field}"].astype(float).tolist()
            )
        micro[metric] = safe_counts(
            int(cases[f"{metric}_tp"].sum()),
            int(cases[f"{metric}_fp"].sum()),
            int(cases[f"{metric}_fn"].sum()),
        )
    for metric in SCALAR_METRICS:
        macro[metric] = distribution(cases[metric].astype(float).tolist())

    prompted = polygons.loc[polygons["prompted"]]
    held_out = polygons.loc[~polygons["prompted"]]
    polygon_summary = {
        "all": {
            "count": len(polygons),
            "recall": distribution(polygons["recall"].tolist()),
        },
        "prompted": {
            "count": len(prompted),
            "recall": distribution(prompted["recall"].tolist()),
            "retrieved_recall_ge_0_1": float((prompted["recall"] >= 0.1).mean()),
            "retrieved_recall_ge_0_5": float((prompted["recall"] >= 0.5).mean()),
            "area_weighted_recall": float(
                prompted["tp"].sum() / max((prompted["tp"] + prompted["fn"]).sum(), 1)
            ),
        },
        "held_out": {
            "count": len(held_out),
            "recall": distribution(held_out["recall"].tolist()),
            "retrieved_recall_ge_0_1": float((held_out["recall"] >= 0.1).mean()),
            "retrieved_recall_ge_0_5": float((held_out["recall"] >= 0.5).mean()),
            "area_weighted_recall": float(
                held_out["tp"].sum() / max((held_out["tp"] + held_out["fn"]).sum(), 1)
            ),
        },
    }
    summary = {
        "timestamp": args.timestamp,
        "status": "complete",
        "case_count": len(cases),
        "annotation_count": len(polygons),
        "prompted_polygon_count": len(prompted),
        "held_out_polygon_count": len(held_out),
        "macro_by_wsi": macro,
        "micro_pooled_pixels": micro,
        "polygon_level": polygon_summary,
        "primary_external_retrieval_metrics": {
            "held_out_polygon_macro_recall": polygon_summary["held_out"]["recall"]["mean"],
            "held_out_retrieved_recall_ge_0_1": polygon_summary["held_out"][
                "retrieved_recall_ge_0_1"
            ],
            "held_out_retrieved_recall_ge_0_5": polygon_summary["held_out"][
                "retrieved_recall_ge_0_5"
            ],
            "held_out_area_weighted_recall": polygon_summary["held_out"][
                "area_weighted_recall"
            ],
            "held_out_censored_dice_macro_wsi": macro[
                "held_out_censored_prompted_region_dice"
            ]["mean"],
            "held_out_censored_precision_macro_wsi": macro[
                "held_out_censored_prompted_region_precision"
            ]["mean"],
        },
        "protocol_note": (
            "Within each WSI, approximately half of annotated TLS polygons provide one "
            "positive prompt each; the other half are held out for retrieval evaluation. "
            "Censored held-out metrics exclude prompted TLS pixels from the evaluation domain."
        ),
        "report_paths": [str(path) for path in report_paths],
    }
    cases.to_parquet(output / f"tls_metrics_by_wsi_{args.timestamp}.parquet", index=False)
    cases.to_csv(output / f"tls_metrics_by_wsi_{args.timestamp}.csv", index=False)
    polygons.to_parquet(
        output / f"tls_metrics_by_polygon_{args.timestamp}.parquet", index=False
    )
    polygons.to_csv(output / f"tls_metrics_by_polygon_{args.timestamp}.csv", index=False)
    (output / f"tls_metrics_summary_{args.timestamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
