#!/usr/bin/env python3
"""Re-aggregate the frozen 1/3/5-positive results on their common cohort.

This is an exact metric replay from persisted per-occurrence TP/FP/FN and
boundary statistics.  It does not rerun model inference or read GT masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.v4.baseline.common import (
    atomic_json,
    new_output_directory,
    sha256_path,
    timestamp,
    validate_episode_manifest,
)


MANIFESTS = {
    1: Path("/nfs-medical3/zyh/v4/baseline/point_1p_1n_manifest_20260724_173500/episode_manifest_20260724_173500.parquet"),
    3: Path("/nfs-medical3/zyh/v4/baseline/point_3p_1n_manifest_20260724_173501/episode_manifest_20260724_173501.parquet"),
    5: Path("/nfs-medical3/zyh/v4/baseline/point_5p_1n_manifest_20260724_173502/episode_manifest_20260724_173502.parquet"),
}

EXTERNAL_RUNS = {
    "SAM": {
        1: Path("/nfs-medical3/zyh/v4/baseline/sam_20260724_000200"),
        3: Path("/nfs-medical3/zyh/v4/baseline/sam_20260724_200000"),
        5: Path("/nfs-medical3/zyh/v4/baseline/sam_20260724_200001"),
    },
    "SAM-Med2D": {
        1: Path("/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_000201"),
        3: Path("/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_200010"),
        5: Path("/nfs-medical3/zyh/v4/baseline/sam_med2d_20260724_200011"),
    },
    "WSI-SAM": {
        1: Path("/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_000202"),
        3: Path("/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_200020"),
        5: Path("/nfs-medical3/zyh/v4/baseline/wsi_sam_20260724_200021"),
    },
}

JOINT_RUNS = {
    1: Path("/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173800/episode_metrics.parquet"),
    3: Path("/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173801/episode_metrics.parquet"),
    5: Path("/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260724_173802/episode_metrics.parquet"),
}

EXPECTED_SOURCE_METRICS = {
    ("SAM", 1): (0.4860059984377517, 0.46490123247843723, 0.3094191216182644, 0.13690138529336765),
    ("SAM", 3): (0.5902590102556224, 0.5804974205547063, 0.4142983899314509, 0.17630055525687402),
    ("SAM", 5): (0.6289818918060294, 0.6242792839640635, 0.4585089886090672, 0.19558022558310276),
    ("SAM-Med2D", 1): (0.21187668939676504, 0.23182517981241366, 0.13793400176491102, 0.09544624500903592),
    ("SAM-Med2D", 3): (0.3177993749753139, 0.33228148396614265, 0.20890452085084285, 0.09800036348968928),
    ("SAM-Med2D", 5): (0.40500265542986436, 0.41488177894892936, 0.27056934892768725, 0.10914008487913741),
    ("WSI-SAM", 1): (0.41536138980634585, 0.4063505741881041, 0.26525240033553654, 0.1353176372611385),
    ("WSI-SAM", 3): (0.5622560872128599, 0.5608003505792803, 0.39763339713039864, 0.16756716702694505),
    ("WSI-SAM", 5): (0.6040811325548539, 0.6057179837344587, 0.4396686764645373, 0.17635170915923457),
    ("Joint model J5", 1): (0.7521589513288406, 0.7418221173508992, 0.5937856290408607, 0.29705328197873737),
    ("Joint model J5", 3): (0.8164109470079082, 0.8158896374899718, 0.691253256616951, 0.3323763244076167),
    ("Joint model J5", 5): (0.8225425916674992, 0.8205688067059566, 0.6978287579114983, 0.3228615285728737),
}

IDENTITY_COLUMNS = (
    "episode_index", "patch_id", "wsi_id", "patient_id", "target_class",
    "x_10x", "y_10x", "width_10x", "height_10x", "wsi_path", "gt_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--limit", type=int, default=885, help="Retain the first N common occurrences; use 1 for smoke.")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/nfs-medical3/zyh/v4/baseline"),
    )
    parser.add_argument("--output-prefix", default="common_885_point_curve")
    return parser.parse_args()


def common_id(value: str) -> str:
    return re.sub(r"_p[135]n1$", "", str(value))


def parse_array(value: Any) -> list[Any]:
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError(f"prompt field is not a list: {value!r}")
    return result


def load_manifests() -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    manifests: dict[int, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for budget, path in MANIFESTS.items():
        frame = pd.read_parquet(path).sort_values("occurrence_order").reset_index(drop=True)
        audits[str(budget)] = validate_episode_manifest(frame, "val")
        expected_rows = {1: 4000, 3: 1261, 5: 885}[budget]
        if len(frame) != expected_rows:
            raise ValueError(f"{budget}+1 manifest has {len(frame)} rows, expected {expected_rows}")
        frame["common_occurrence_id"] = frame["occurrence_id"].map(common_id)
        if frame["common_occurrence_id"].duplicated().any():
            raise ValueError(f"{budget}+1 manifest has duplicate common occurrence IDs")
        manifests[budget] = frame

    common = manifests[5].set_index("common_occurrence_id", drop=False)
    for budget in (1, 3):
        candidate = manifests[budget].set_index("common_occurrence_id", drop=False)
        if not common.index.isin(candidate.index).all():
            raise ValueError(f"5+1 cohort is not nested in {budget}+1 cohort")
        aligned = candidate.loc[common.index]
        for column in IDENTITY_COLUMNS:
            if aligned[column].astype(str).tolist() != common[column].astype(str).tolist():
                raise ValueError(f"identity mismatch for {budget}+1 column {column}")
        for key in common.index:
            small = aligned.loc[key]
            large = common.loc[key]
            if parse_array(small.positive_points_10x) != parse_array(large.positive_points_10x)[:budget]:
                raise ValueError(f"positive point prefix mismatch for {key}, budget={budget}")
            if parse_array(small.source_region_ids) != parse_array(large.source_region_ids)[:budget]:
                raise ValueError(f"positive source-region prefix mismatch for {key}, budget={budget}")
            if parse_array(small.negative_points_10x) != parse_array(large.negative_points_10x):
                raise ValueError(f"negative point mismatch for {key}, budget={budget}")
            if parse_array(small.negative_source_region_ids) != parse_array(large.negative_source_region_ids):
                raise ValueError(f"negative source-region mismatch for {key}, budget={budget}")
    return manifests, audits


def normalize_external(run_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    stamp = run_dir.name.rsplit("_", 2)[-2] + "_" + run_dir.name.rsplit("_", 1)[-1]
    paths = sorted((run_dir / f"shards_{stamp}").glob(f"episodes_rank*_shard*_{stamp}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no metric shards under {run_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame.sort_values("occurrence_order").reset_index(drop=True)
    if len(frame) != len(manifest):
        raise ValueError(f"{run_dir} has {len(frame)} rows, expected {len(manifest)}")
    if not np.array_equal(frame["occurrence_order"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"non-contiguous occurrence order in {run_dir}")
    for column in ("episode_index", "patch_id", "wsi_id", "patient_id", "target_class"):
        if frame[column].astype(str).tolist() != manifest[column].astype(str).tolist():
            raise ValueError(f"metric/manifest identity mismatch in {run_dir}: {column}")
    if set(frame["status"].astype(str)) != {"completed"}:
        raise ValueError(f"non-completed rows found in {run_dir}")
    result = frame[[
        "occurrence_order", "target_class", "tp", "fp", "fn", "episode_dice",
        "boundary_f1", "boundary_evaluable", "status",
    ]].copy()
    result["common_occurrence_id"] = manifest["common_occurrence_id"].to_numpy()
    return result


def normalize_joint(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("episode_index").reset_index(drop=True)
    if len(frame) != len(manifest) or not np.array_equal(frame["episode_index"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"joint metric rows do not align positionally with {path}")
    for column in ("patch_id", "wsi_id", "target_class"):
        if frame[column].astype(str).tolist() != manifest[column].astype(str).tolist():
            raise ValueError(f"joint metric/manifest identity mismatch in {path}: {column}")
    result = pd.DataFrame({
        "occurrence_order": np.arange(len(frame)),
        "common_occurrence_id": manifest["common_occurrence_id"].to_numpy(),
        "target_class": frame["target_class"].to_numpy(),
        "tp": frame["joint_pixel_tp"].to_numpy(),
        "fp": frame["joint_pixel_fp"].to_numpy(),
        "fn": frame["joint_pixel_fn"].to_numpy(),
        "episode_dice": frame["joint_pixel_dice"].to_numpy(),
        "boundary_f1": frame["joint_boundary_f1"].to_numpy(),
        "boundary_evaluable": frame["joint_boundary_evaluable"].to_numpy(),
        "status": "completed",
    })
    return result


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or set(frame["status"].astype(str)) != {"completed"}:
        raise ValueError("metric summary requires non-empty completed rows")
    tp, fp, fn = (int(frame[column].sum()) for column in ("tp", "fp", "fn"))
    denominator = 2 * tp + fp + fn
    by_class = frame.groupby("target_class", sort=True)[["tp", "fp", "fn"]].sum()
    class_dice = 2 * by_class["tp"] / (2 * by_class["tp"] + by_class["fp"] + by_class["fn"])
    class_iou = by_class["tp"] / (by_class["tp"] + by_class["fp"] + by_class["fn"])
    return {
        "episodes": int(len(frame)),
        "unique_classes": int(frame["target_class"].nunique()),
        "global_dice": float(2 * tp / denominator),
        "class_macro_dice": float(class_dice.mean()),
        "class_macro_iou": float(class_iou.mean()),
        "boundary_f1_2px": float(frame["boundary_f1"].mean(skipna=True)),
        "boundary_evaluable": int(frame["boundary_f1"].notna().sum()),
        "macro_episode_dice": float(frame["episode_dice"].mean(skipna=True)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def markdown_table(rows: pd.DataFrame) -> str:
    display = rows.copy()
    for column in ("global_dice", "class_macro_dice", "class_macro_iou", "boundary_f1_2px"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display = display.rename(columns={
        "budget": "提示协议", "episodes": "Occurrences", "method": "方法",
        "global_dice": "全局 Dice", "class_macro_dice": "类别宏 Dice",
        "class_macro_iou": "mIoU", "boundary_f1_2px": "Boundary F1",
    })
    columns = [str(column) for column in display.columns]
    values = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in values)) for i in range(len(columns))]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in values]
    return "\n".join([header, rule, *body])


def main() -> None:
    args = parse_args()
    stamp = timestamp(args.timestamp)
    if not 1 <= args.limit <= 885:
        raise ValueError("--limit must be between 1 and 885")
    manifests, manifest_audits = load_manifests()
    common_ids = manifests[5]["common_occurrence_id"].tolist()[: args.limit]
    common_set = set(common_ids)
    source_replay: dict[str, Any] = {}
    long_frames: list[pd.DataFrame] = []

    for method, runs in EXTERNAL_RUNS.items():
        for budget, run_dir in runs.items():
            frame = normalize_external(run_dir, manifests[budget])
            replay = metric_summary(frame)
            observed = tuple(replay[key] for key in ("global_dice", "class_macro_dice", "class_macro_iou", "boundary_f1_2px"))
            if not np.allclose(observed, EXPECTED_SOURCE_METRICS[(method, budget)], atol=1e-12, rtol=0):
                raise ValueError(f"published source-metric replay failed for {method} {budget}+1: {observed}")
            source_replay[f"{method}|{budget}+1"] = replay
            selected = frame[frame["common_occurrence_id"].isin(common_set)].set_index("common_occurrence_id").loc[common_ids].reset_index()
            selected["method"] = method
            selected["positive_points"] = budget
            long_frames.append(selected)

    for budget, path in JOINT_RUNS.items():
        frame = normalize_joint(path, manifests[budget])
        replay = metric_summary(frame)
        observed = tuple(replay[key] for key in ("global_dice", "class_macro_dice", "class_macro_iou", "boundary_f1_2px"))
        if not np.allclose(observed, EXPECTED_SOURCE_METRICS[("Joint model J5", budget)], atol=1e-12, rtol=0):
            raise ValueError(f"published source-metric replay failed for J5 {budget}+1: {observed}")
        source_replay[f"Joint model J5|{budget}+1"] = replay
        selected = frame[frame["common_occurrence_id"].isin(common_set)].set_index("common_occurrence_id").loc[common_ids].reset_index()
        selected["method"] = "Joint model J5"
        selected["positive_points"] = budget
        long_frames.append(selected)

    long_frame = pd.concat(long_frames, ignore_index=True)
    expected_long_rows = 4 * 3 * args.limit
    if len(long_frame) != expected_long_rows:
        raise ValueError(f"common-cohort metric rows={len(long_frame)}, expected={expected_long_rows}")
    expected_alignment = common_ids * 12
    if long_frame["common_occurrence_id"].tolist() != expected_alignment:
        raise ValueError("common-cohort row alignment failed")

    summary_rows: list[dict[str, Any]] = []
    for budget in (1, 3, 5):
        for method in ("SAM", "SAM-Med2D", "WSI-SAM", "Joint model J5"):
            part = long_frame[(long_frame["method"] == method) & (long_frame["positive_points"] == budget)]
            metrics = metric_summary(part)
            summary_rows.append({
                "budget": f"{budget}+1", "episodes": metrics["episodes"], "method": method,
                "global_dice": metrics["global_dice"], "class_macro_dice": metrics["class_macro_dice"],
                "class_macro_iou": metrics["class_macro_iou"], "boundary_f1_2px": metrics["boundary_f1_2px"],
            })
    summary_frame = pd.DataFrame(summary_rows)

    common_manifest = manifests[5].iloc[: args.limit].copy()
    cohort_audit = {
        "occurrences": int(len(common_manifest)),
        "unique_episode_indices": int(common_manifest["episode_index"].nunique()),
        "unique_patches": int(common_manifest["patch_id"].nunique()),
        "unique_wsi": int(common_manifest["wsi_id"].nunique()),
        "unique_patients": int(common_manifest["patient_id"].nunique()),
        "target_class_counts": {str(k): int(v) for k, v in common_manifest["target_class"].value_counts().sort_index().items()},
    }
    output = new_output_directory(args.output_root, args.output_prefix, stamp)
    cohort_path = output / f"common_cohort_{stamp}.parquet"
    long_path = output / f"episode_metrics_long_{stamp}.parquet"
    table_path = output / f"common_curve_metrics_{stamp}.csv"
    common_manifest.to_parquet(cohort_path, index=False)
    long_frame.to_parquet(long_path, index=False)
    summary_frame.to_csv(table_path, index=False)

    inputs = {
        "manifests": {str(k): {"path": str(v), "sha256": sha256_path(v)} for k, v in MANIFESTS.items()},
        "external_runs": {method: {str(k): str(v) for k, v in runs.items()} for method, runs in EXTERNAL_RUNS.items()},
        "joint_metrics": {str(k): {"path": str(v), "sha256": sha256_path(v)} for k, v in JOINT_RUNS.items()},
    }
    audit = {
        "timestamp": stamp,
        "mode": "exact_reaggregation_from_persisted_per_occurrence_counts_no_inference",
        "limit": args.limit,
        "manifest_audits": manifest_audits,
        "common_cohort": cohort_audit,
        "prompt_contract": "same occurrences/classes/GT; positive prompts are frozen prefixes of the 5-point list; identical frozen negative point",
        "source_metric_replay": source_replay,
        "source_metric_replay_tolerance": 1e-12,
        "inputs": inputs,
        "outputs": {"cohort": str(cohort_path), "episode_metrics": str(long_path), "metrics_csv": str(table_path)},
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": [sys.executable, *sys.argv],
        "status": "passed",
    }
    audit_path = output / f"audit_{stamp}.json"
    atomic_json(audit_path, audit)

    report_path = output / f"report_{stamp}.md"
    report = [
        f"# 固定共同子集点数曲线（{stamp}）", "",
        "本报告在 5+1 可用的冻结 occurrence 子集上，对已有逐 occurrence 推理统计进行精确重汇总；未重新运行模型推理。",
        "三档实验使用相同 occurrence、目标类别、GT 与负点，正点分别取同一冻结 5 点序列的前 1、3、5 个。", "",
        f"共同子集：{cohort_audit['occurrences']} occurrences，{cohort_audit['unique_episode_indices']} 个唯一 episode index，"
        f"{cohort_audit['unique_patches']} 个唯一 patch，{cohort_audit['unique_wsi']} 个 WSI。", "",
        markdown_table(summary_frame), "",
        "全局 Dice 按所有 occurrence 的 TP/FP/FN 像素池化；类别宏 Dice 与 mIoU 先在每个目标类别内池化，再对类别等权平均；Boundary F1 为 2 像素容差的 occurrence 宏平均。", "",
        f"审计：`{audit_path}`", f"逐 occurrence 统计：`{long_path}`", f"机器可读表：`{table_path}`", "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "report": str(report_path), "audit": str(audit_path),
        "cohort": cohort_audit, "metrics": summary_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
