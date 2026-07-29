#!/usr/bin/env python3
"""Derive paper-facing J5 WSI metrics from an audited candidate metric table."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.baseline.tools.build_multiclass_wsi_summary import (
    CLASS_ZH,
    metric_fields,
)


METRICS = ["dice", "iou", "precision", "recall"]
TASK_KEYS = ["wsi_id", "target_class", "target_class_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(group: pd.DataFrame) -> dict[str, float | int]:
    pooled = metric_fields(
        int(group["gt_positive_pixels"].sum()),
        int(group["prediction_positive_pixels"].sum()),
        int(group["intersection_pixels"].sum()),
    )
    return {
        **{f"macro_{metric}": float(group[metric].mean()) for metric in METRICS},
        **{f"micro_{metric}": float(pooled[metric]) for metric in METRICS},
        "gt_positive_pixels": int(pooled["gt_positive_pixels"]),
        "prediction_positive_pixels": int(pooled["prediction_positive_pixels"]),
        "intersection_pixels": int(pooled["intersection_pixels"]),
    }


def validate(frame: pd.DataFrame, source_audit: dict[str, object]) -> None:
    required = {
        *TASK_KEYS,
        "candidate_index",
        "candidate_status",
        "gt_positive_pixels",
        "prediction_positive_pixels",
        "intersection_pixels",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate metric columns missing: {missing}")
    if len(frame) != 300 or frame.duplicated(TASK_KEYS + ["candidate_index"]).any():
        raise ValueError("expected 300 unique WSI-class-candidate rows")
    task_sizes = frame.groupby(TASK_KEYS).size()
    if len(task_sizes) != 60 or not task_sizes.eq(5).all():
        raise ValueError("expected 60 tasks with five candidates each")
    candidate_sets = frame.groupby(TASK_KEYS)["candidate_index"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if not candidate_sets.map(lambda value: value == (0, 1, 2, 3, 4)).all():
        raise ValueError("candidate indices must be 0..4 for every task")
    if int(frame["candidate_status"].eq("complete").sum()) != 299:
        raise ValueError("expected 299 completed candidates")
    if int(frame["candidate_status"].eq("abstained_prompt_conflict").sum()) != 1:
        raise ValueError("expected one prompt-conflict abstention")
    if source_audit.get("abstention_policy") != "intention_to_evaluate_as_empty_prediction":
        raise ValueError("source audit does not declare the frozen abstention policy")
    if source_audit.get("semantic_audit_pass") is not True:
        raise ValueError("source prompt semantic audit did not pass")
    if not frame[METRICS].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("metric outside [0, 1]")
    gt_unique = frame.groupby(TASK_KEYS)["gt_positive_pixels"].nunique()
    if not gt_unique.eq(1).all():
        raise ValueError("GT positive-pixel count differs across candidates")
    for row in frame.itertuples(index=False):
        expected = metric_fields(
            int(row.gt_positive_pixels),
            int(row.prediction_positive_pixels),
            int(row.intersection_pixels),
        )
        if any(
            not np.isclose(float(getattr(row, metric)), float(expected[metric]), atol=1e-12)
            for metric in METRICS
        ):
            raise ValueError(
                f"stored metric mismatch for {row.wsi_id}/"
                f"{row.target_class_name}/candidate_{int(row.candidate_index):02d}"
            )


def main() -> None:
    args = parse_args()
    output = args.output_root / f"j5_quality_paper_metrics_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.candidate_metrics)
    validate(frame, source_audit)

    repeat_rows = []
    for candidate_index, group in frame.groupby("candidate_index", sort=True):
        repeat_rows.append(
            {
                "repeat": int(candidate_index) + 1,
                "candidate_index": int(candidate_index),
                "n_tasks": len(group),
                **aggregate(group),
            }
        )
    repeats = pd.DataFrame(repeat_rows)

    repeat_summary = {}
    for prefix in ("macro", "micro"):
        for metric in METRICS:
            values = repeats[f"{prefix}_{metric}"]
            repeat_summary[f"{prefix}_{metric}_mean"] = float(values.mean())
            repeat_summary[f"{prefix}_{metric}_sd"] = float(values.std(ddof=1))

    class_repeat = (
        frame.groupby(
            ["target_class", "target_class_name", "candidate_index"], as_index=False
        )[METRICS]
        .mean()
        .sort_values(["target_class", "candidate_index"])
    )
    class_repeat_summary = (
        class_repeat.groupby(["target_class", "target_class_name"], as_index=False)[METRICS]
        .agg(["mean", "std"])
    )
    class_repeat_summary.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        for column in class_repeat_summary.columns.to_flat_index()
    ]

    best = (
        frame.sort_values(
            TASK_KEYS + ["dice", "candidate_index"],
            ascending=[True, True, True, False, True],
        )
        .groupby(TASK_KEYS, as_index=False, sort=False)
        .head(1)
        .sort_values(["target_class", "wsi_id"])
    )
    if len(best) != 60 or not best["candidate_status"].eq("complete").all():
        raise RuntimeError("invalid best-of-five task selection")
    best_overall = aggregate(best)
    best_class = (
        best.groupby(["target_class", "target_class_name"], as_index=False)[METRICS]
        .mean()
        .sort_values("target_class")
    )

    frame.to_parquet(output / f"candidate_metrics_audited_{args.timestamp}.parquet", index=False)
    repeats.to_parquet(output / f"per_repeat_metrics_{args.timestamp}.parquet", index=False)
    class_repeat.to_parquet(
        output / f"per_class_repeat_metrics_{args.timestamp}.parquet", index=False
    )
    class_repeat_summary.to_parquet(
        output / f"per_class_repeat_summary_{args.timestamp}.parquet", index=False
    )
    best.to_parquet(output / f"best_of_five_task_metrics_{args.timestamp}.parquet", index=False)
    best_class.to_parquet(
        output / f"best_of_five_class_metrics_{args.timestamp}.parquet", index=False
    )

    audit = {
        "status": "complete",
        "timestamp": args.timestamp,
        "candidate_metrics": str(args.candidate_metrics),
        "candidate_metrics_sha256": sha256(args.candidate_metrics),
        "source_audit": str(args.source_audit),
        "source_audit_sha256": sha256(args.source_audit),
        "candidate_selection_dice_blind": True,
        "abstention_policy": source_audit["abstention_policy"],
        "n_tasks": 60,
        "n_candidates": 300,
        "n_complete_candidates": 299,
        "n_abstained_candidates": 1,
        "repeat_summary": repeat_summary,
        "best_of_five_gt_oracle": best_overall,
        "reporting_boundary": (
            "Five-repeat mean±SD is outcome-blind after prompt generation. "
            "Best-of-five uses GT Dice after inference and is an oracle upper bound."
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# J5 质量受控 WSI 论文指标",
        "",
        "## Dice-blind 五次独立结果",
        "",
        "| 次数 | Macro Dice | Macro IoU | Macro Precision | Macro Recall | Micro Dice |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repeats.itertuples(index=False):
        lines.append(
            f"| {row.repeat} | {row.macro_dice:.4f} | {row.macro_iou:.4f} | "
            f"{row.macro_precision:.4f} | {row.macro_recall:.4f} | "
            f"{row.micro_dice:.4f} |"
        )
    lines.extend(
        [
            "",
            "五次均值 ± 样本标准差：",
            "",
            "| Macro Dice | Macro IoU | Macro Precision | Macro Recall | Micro Dice |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {repeat_summary['macro_dice_mean']:.4f} ± "
                f"{repeat_summary['macro_dice_sd']:.4f} | "
                f"{repeat_summary['macro_iou_mean']:.4f} ± "
                f"{repeat_summary['macro_iou_sd']:.4f} | "
                f"{repeat_summary['macro_precision_mean']:.4f} ± "
                f"{repeat_summary['macro_precision_sd']:.4f} | "
                f"{repeat_summary['macro_recall_mean']:.4f} ± "
                f"{repeat_summary['macro_recall_sd']:.4f} | "
                f"{repeat_summary['micro_dice_mean']:.4f} ± "
                f"{repeat_summary['micro_dice_sd']:.4f} |"
            ),
            "",
            "## 文档原协议：GT best-of-five oracle 上界",
            "",
            "| Macro Dice | Macro IoU | Macro Precision | Macro Recall | Micro Dice |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {best_overall['macro_dice']:.4f} | "
                f"{best_overall['macro_iou']:.4f} | "
                f"{best_overall['macro_precision']:.4f} | "
                f"{best_overall['macro_recall']:.4f} | "
                f"{best_overall['micro_dice']:.4f} |"
            ),
            "",
            "## 十二类 GT best-of-five Dice",
            "",
            "| 组织类别 | Dice |",
            "|---|---:|",
        ]
    )
    for row in best_class.itertuples(index=False):
        lines.append(
            f"| {CLASS_ZH.get(row.target_class_name, row.target_class_name)} | "
            f"{row.dice:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 质量受控提示在推理前确定，不读取预测 Dice。",
            "- 五次均值 ± 标准差是 outcome-blind 主结果。",
            "- best-of-five 在推理后读取 GT Dice 选优，只能表述为 oracle 上界。",
            "- 1/300 个候选因正负提示映射到同一 learned region 而弃权；不补抽，按空预测计入五次重复。",
            "",
        ]
    )
    (output / "paper_metrics_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "paper_metrics_complete", "output": str(output), **audit}))


if __name__ == "__main__":
    main()
