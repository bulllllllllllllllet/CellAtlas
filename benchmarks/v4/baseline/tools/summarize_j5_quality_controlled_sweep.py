#!/usr/bin/env python3
"""Summarize five quality-controlled J5 repeats without outcome-based filtering."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.baseline.tools.build_multiclass_wsi_summary import (
    CLASS_ZH,
    candidate_metrics,
    load_class_contract,
    metric_fields,
)


METRICS = ["dice", "iou", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--limit-tasks",
        type=int,
        help="evaluate only the first N manifest tasks for a retained smoke run",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        help="evaluate one zero-based manifest task for a retained smoke run",
    )
    return parser.parse_args()


def mean_sd(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def render_value(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def main() -> None:
    args = parse_args()
    output = args.output_root / f"j5_quality_summary_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    candidate_meta_path = args.candidate_root / "metadata.json"
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    if candidate_meta.get("dice_blind") is not True:
        raise ValueError("candidate selection is not marked Dice-blind")
    candidate_manifest = pd.read_parquet(candidate_meta["combined_manifest"])
    if len(candidate_manifest) != 300:
        raise ValueError(f"expected 300 prompt candidates, got {len(candidate_manifest)}")
    if not candidate_manifest[["positive_audit", "negative_audit"]].all().all():
        raise ValueError("prompt semantic audit failed")

    run_meta_path = args.sweep_root / "metadata.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_meta["run_manifest"])
    if len(run_manifest) != 60:
        raise ValueError(f"expected 60 WSI-class tasks, got {len(run_manifest)}")
    if not (args.sweep_root / "complete.json").is_file():
        raise RuntimeError("sweep is not marked complete")
    if args.limit_tasks is not None and args.task_index is not None:
        raise ValueError("--limit-tasks and --task-index are mutually exclusive")
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(run_manifest):
            raise ValueError("--task-index must be in [0, 59]")
        run_manifest = run_manifest.iloc[[args.task_index]].copy()
    elif args.limit_tasks is not None:
        if args.limit_tasks <= 0 or args.limit_tasks > len(run_manifest):
            raise ValueError("--limit-tasks must be in [1, 60]")
        run_manifest = run_manifest.head(args.limit_tasks).copy()

    _, class_rgbs = load_class_contract()
    rows: list[dict[str, object]] = []
    for task_no, task in enumerate(run_manifest.itertuples(index=False), start=1):
        inference_dir = (
            args.sweep_root
            / "runs"
            / str(task.target_class_name)
            / str(task.wsi_id)
            / f"j5_multi_prompt_wsi_{run_meta['timestamp']}"
        )
        metadata_path = inference_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") != "complete"
            or int(metadata.get("requested_candidate_count", -1)) != 5
        ):
            raise RuntimeError(f"incomplete five-candidate result: {metadata_path}")
        summaries = {
            int(item["candidate_index"]): item for item in metadata["candidates"]
        }
        if len(summaries) != int(metadata.get("decoded_candidate_count", -1)):
            raise RuntimeError(f"decoded candidate count mismatch: {metadata_path}")
        if not summaries:
            raise RuntimeError(f"all five candidates abstained: {metadata_path}")
        shape = tuple(map(int, next(iter(summaries.values()))["output_shape_10x"]))
        complete_indices = sorted(summaries)
        mask_paths = [Path(summaries[index]["mask_tiff"]) for index in complete_indices]
        if not all(path.is_file() for path in mask_paths):
            raise FileNotFoundError(f"missing completed-candidate TIFF under {inference_dir}")
        completed_metrics = candidate_metrics(
            Path(task.gt_path),
            class_rgbs[str(task.target_class_name)],
            shape,
            mask_paths,
        )
        metrics_by_index = dict(zip(complete_indices, completed_metrics, strict=True))
        gt_count = int(completed_metrics[0]["gt_positive_pixels"])
        prompt_rows = candidate_manifest.loc[
            candidate_manifest["wsi_id"].eq(task.wsi_id)
            & candidate_manifest["target_class_name"].eq(task.target_class_name)
        ].set_index("candidate_index")
        if set(prompt_rows.index) != set(range(5)):
            raise RuntimeError(f"missing prompt audit rows for {task.wsi_id}/{task.target_class_name}")
        for candidate_index in range(5):
            prompt = prompt_rows.loc[candidate_index]
            candidate_metadata_path = (
                inference_dir / f"candidate_{candidate_index:02d}" / "metadata.json"
            )
            candidate_metadata = json.loads(
                candidate_metadata_path.read_text(encoding="utf-8")
            )
            status = str(candidate_metadata.get("status"))
            if candidate_index in metrics_by_index:
                if status != "complete":
                    raise RuntimeError(
                        f"candidate summary/status mismatch: {candidate_metadata_path}"
                    )
                values = metrics_by_index[candidate_index]
                mask_tiff: str | None = str(
                    Path(candidate_metadata["mask_tiff"])
                )
            else:
                if (
                    status != "abstained_prompt_conflict"
                    or candidate_metadata.get("mask_returned") is not False
                ):
                    raise RuntimeError(
                        f"unsupported missing-candidate status: {candidate_metadata_path}"
                    )
                # Formal intention-to-evaluate scoring: an abstention remains an
                # abstention (no replacement mask) and receives the metrics of an
                # empty prediction against the known-positive target class.
                values = metric_fields(gt_count, 0, 0)
                mask_tiff = None
            rows.append(
                {
                    "wsi_id": str(task.wsi_id),
                    "target_class": int(task.target_class),
                    "target_class_name": str(task.target_class_name),
                    "candidate_index": candidate_index,
                    "target_fraction": float(prompt["target_fraction"]),
                    "quality_threshold": float(prompt["quality_threshold"]),
                    "selection_rule": str(prompt["selection_rule"]),
                    "positive_audit": bool(prompt["positive_audit"]),
                    "negative_audit": bool(prompt["negative_audit"]),
                    "prompt_json": str(prompt["prompt_json"]),
                    "candidate_status": status,
                    "abstention_scoring": (
                        "empty_prediction"
                        if status == "abstained_prompt_conflict"
                        else "not_applicable"
                    ),
                    "mask_tiff": mask_tiff,
                    **values,
                }
            )
        print(
            json.dumps(
                {
                    "event": "task_metrics_complete",
                    "task": task_no,
                    "total": len(run_manifest),
                    "wsi_id": task.wsi_id,
                    "class": task.target_class_name,
                }
            ),
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values(
        ["candidate_index", "target_class", "wsi_id"]
    )
    expected_candidates = len(run_manifest) * 5
    if len(frame) != expected_candidates or frame.duplicated(
        ["wsi_id", "target_class_name", "candidate_index"]
    ).any():
        raise RuntimeError(
            f"candidate metric table is not {expected_candidates} unique rows"
        )

    repeat_rows: list[dict[str, object]] = []
    for repeat, group in frame.groupby("candidate_index", sort=True):
        micro = metric_fields(
            int(group["gt_positive_pixels"].sum()),
            int(group["prediction_positive_pixels"].sum()),
            int(group["intersection_pixels"].sum()),
        )
        repeat_rows.append(
            {
                "repeat": int(repeat) + 1,
                "candidate_index": int(repeat),
                "n_tasks": len(group),
                **{f"macro_{metric}": float(group[metric].mean()) for metric in METRICS},
                **{f"micro_{metric}": float(micro[metric]) for metric in METRICS},
                "gt_positive_pixels": int(micro["gt_positive_pixels"]),
                "prediction_positive_pixels": int(micro["prediction_positive_pixels"]),
                "intersection_pixels": int(micro["intersection_pixels"]),
            }
        )
    repeats = pd.DataFrame(repeat_rows)

    class_repeat = (
        frame.groupby(["target_class", "target_class_name", "candidate_index"], as_index=False)[
            METRICS
        ]
        .mean()
        .sort_values(["target_class", "candidate_index"])
    )
    class_summary_rows: list[dict[str, object]] = []
    for (class_id, class_name), group in class_repeat.groupby(
        ["target_class", "target_class_name"], sort=True
    ):
        record: dict[str, object] = {
            "target_class": int(class_id),
            "target_class_name": str(class_name),
            "n_repeats": len(group),
        }
        for metric in METRICS:
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd(group[metric])
        class_summary_rows.append(record)
    class_summary = pd.DataFrame(class_summary_rows)

    overall: dict[str, float] = {}
    for prefix in ("macro", "micro"):
        for metric in METRICS:
            overall[f"{prefix}_{metric}_mean"], overall[f"{prefix}_{metric}_sd"] = mean_sd(
                repeats[f"{prefix}_{metric}"]
            )

    frame.to_parquet(output / f"candidate_metrics_{args.timestamp}.parquet", index=False)
    repeats.to_parquet(output / f"per_repeat_metrics_{args.timestamp}.parquet", index=False)
    class_repeat.to_parquet(
        output / f"per_class_repeat_metrics_{args.timestamp}.parquet", index=False
    )
    class_summary.to_parquet(
        output / f"per_class_summary_{args.timestamp}.parquet", index=False
    )

    audit = {
        "status": "complete",
        "timestamp": args.timestamp,
        "sweep_root": str(args.sweep_root),
        "candidate_root": str(args.candidate_root),
        "candidate_selection_dice_blind": True,
        "abstention_policy": "intention_to_evaluate_as_empty_prediction",
        "selection_rule": sorted(frame["selection_rule"].unique().tolist()),
        "n_tasks": int(frame[["wsi_id", "target_class_name"]].drop_duplicates().shape[0]),
        "n_candidates": len(frame),
        "n_repeats": len(repeats),
        "n_complete_candidates": int(frame["candidate_status"].eq("complete").sum()),
        "n_abstained_candidates": int(
            frame["candidate_status"].eq("abstained_prompt_conflict").sum()
        ),
        "semantic_audit_pass": bool(frame[["positive_audit", "negative_audit"]].all().all()),
        "minimum_target_fraction": float(frame["target_fraction"].min()),
        "overall": overall,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 质量控制后五次 WSI 实验汇总",
        "",
        (
            "候选提示在推理前按目标组织占比筛选：每个 WSI–类别仅从目标占比位于"
            "前 25% 且绝对占比不低于 0.1% 的 Patch 中选取，再进行空间分散采样。"
            "该规则不读取预测结果或 Dice，因此不是事后删除低分实验。"
        ),
        "",
        "## 五次独立候选结果",
        "",
        "| 次数 | Macro Dice | Macro IoU | Macro Precision | Macro Recall | Micro Dice |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repeats.itertuples(index=False):
        lines.append(
            f"| {row.repeat} | {row.macro_dice:.4f} | {row.macro_iou:.4f} | "
            f"{row.macro_precision:.4f} | {row.macro_recall:.4f} | {row.micro_dice:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 五次均值 ± 样本标准差",
            "",
            "| 汇总方式 | Dice | IoU | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
            "| Macro（60 个 WSI–类别任务等权） | "
            + " | ".join(
                render_value(overall[f"macro_{metric}_mean"], overall[f"macro_{metric}_sd"])
                for metric in METRICS
            )
            + " |",
            "| Micro（像素计数汇总） | "
            + " | ".join(
                render_value(overall[f"micro_{metric}_mean"], overall[f"micro_{metric}_sd"])
                for metric in METRICS
            )
            + " |",
            "",
            "## 分类别五次均值 ± 样本标准差",
            "",
            "| 类别 | Dice | IoU | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_summary.itertuples(index=False):
        lines.append(
            f"| {CLASS_ZH.get(row.target_class_name, row.target_class_name)} | "
            + " | ".join(
                render_value(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sd"))
                for metric in METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 完整性审计",
            "",
            f"- 任务数：{audit['n_tasks']}；候选结果数：{audit['n_candidates']}；重复数：5。",
            (
                f"- 完整返回：{audit['n_complete_candidates']}；提示冲突弃权："
                f"{audit['n_abstained_candidates']}。弃权不补抽，按空预测纳入"
                " intention-to-evaluate 指标。"
            ),
            f"- 正/负提示语义审计：{'通过' if audit['semantic_audit_pass'] else '失败'}。",
            f"- 入选 Patch 的最小目标占比：{audit['minimum_target_fraction']:.6f}。",
            "- 未按 Dice、IoU 或任何预测后指标删除、替换或挑选运行。",
            "",
        ]
    )
    (output / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "summary_complete", "output": str(output), **overall}), flush=True)


if __name__ == "__main__":
    main()
