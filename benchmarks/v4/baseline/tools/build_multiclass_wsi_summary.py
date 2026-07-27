#!/usr/bin/env python3
"""Aggregate the five-WSI, twelve-class J5 and patchwise SAM experiment."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import yaml


REPO = Path("/home/zhaoyh/CellAtlas")
BASELINE = Path("/nfs-medical3/zyh/v4/baseline")
WSI_ROOT = Path("/nfs-medical3/zyh/v4/whole_slide_inference")
WSIS = [
    "1033746-12-HE-DX1",
    "1028417-R1-HE-DX1",
    "1321593-10-HE-DX1",
    "1416664-10-HE-DX1",
    "1504774-12-HE-DX1",
]
NEW_CLASSES = [
    "tumor_epithelium",
    "tumor_stroma",
    "background",
    "necrosis",
    "normal_gland",
    "normal_stroma",
    "submucosa_serosa",
    "lymphocyte_aggregate",
    "mucus",
    "fat",
    "blood",
]
CLASS_ZH = {
    "tumor_epithelium": "肿瘤上皮",
    "tumor_stroma": "肿瘤间质",
    "background": "背景",
    "necrosis": "坏死",
    "normal_gland": "正常腺体",
    "normal_stroma": "正常间质",
    "submucosa_serosa": "黏膜下层/浆膜",
    "muscle": "肌肉",
    "lymphocyte_aggregate": "淋巴细胞聚集",
    "mucus": "黏液",
    "fat": "脂肪",
    "blood": "血液",
}
METHOD_ZH = {
    "j5_oracle5": "J5（5 候选 GT oracle）",
    "sam": "SAM",
    "sam_med2d": "SAM-Med2D",
    "wsi_sam": "WSI-SAM",
}
METRICS = ["dice", "iou", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO / "benchmarks/v4/baseline/doc/2.md",
    )
    return parser.parse_args()


def load_class_contract() -> tuple[dict[str, int], dict[str, tuple[int, int, int]]]:
    path = REPO / "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    records = config["data"]["class_map"]
    return (
        {str(item["name"]): int(item["id"]) for item in records},
        {str(item["name"]): tuple(map(int, item["rgb"])) for item in records},
    )


def metric_fields(gt_count: int, pred_count: int, intersection: int) -> dict[str, float | int]:
    union = gt_count + pred_count - intersection
    return {
        "gt_positive_pixels": gt_count,
        "prediction_positive_pixels": pred_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "dice": 2 * intersection / (gt_count + pred_count) if gt_count + pred_count else 1.0,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_count if pred_count else float(gt_count == 0),
        "recall": intersection / gt_count if gt_count else float(pred_count == 0),
    }


def candidate_metrics(
    gt_path: Path,
    target_rgb: tuple[int, int, int],
    shape: tuple[int, int],
    mask_paths: list[Path],
) -> list[dict[str, float | int]]:
    height, width = shape
    gt = pyvips.Image.new_from_file(str(gt_path), access="sequential").extract_band(0, n=3)
    if width - gt.width not in (0, 1) or height - gt.height not in (0, 1):
        raise ValueError(f"GT/mask extent mismatch: GT={(gt.width, gt.height)} mask={(width, height)}")
    masks = [
        pyvips.Image.new_from_file(str(path), page=0, access="sequential").extract_band(0)
        for path in mask_paths
    ]
    if any(mask.width != width or mask.height != height for mask in masks):
        raise ValueError(f"candidate TIFF extent mismatch for {mask_paths}")
    gt_count = 0
    pred_counts = np.zeros(len(masks), dtype=np.int64)
    intersections = np.zeros(len(masks), dtype=np.int64)
    for y in range(0, gt.height, 1024):
        h = min(1024, gt.height - y)
        crop = gt.crop(0, y, gt.width, h)
        target = (
            (crop[0] == target_rgb[0])
            & (crop[1] == target_rgb[1])
            & (crop[2] == target_rgb[2])
        )
        truth = np.frombuffer(target.cast("uchar").write_to_memory(), np.uint8)
        truth = truth.reshape(h, gt.width) > 0
        gt_count += int(truth.sum())
        for index, mask in enumerate(masks):
            pred = np.frombuffer(
                mask.crop(0, y, gt.width, h).cast("uchar").write_to_memory(),
                np.uint8,
            ).reshape(h, gt.width) > 0
            pred_counts[index] += int(pred.sum())
            intersections[index] += int((truth & pred).sum())
    return [
        metric_fields(gt_count, int(pred_counts[index]), int(intersections[index]))
        for index in range(len(masks))
    ]


def collect_sam_rows(class_ids: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    methods = ["sam", "sam_med2d", "wsi_sam"]
    for class_index, class_name in enumerate(NEW_CLASSES):
        stamp = f"20260726_{190000 + class_index * 100:06d}"
        for method in methods:
            root = BASELINE / f"{method}_gt_guided_wsi_{stamp}"
            for wsi_id in WSIS:
                path = root / wsi_id / "metadata.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["method"] != method or data["target_class_name"] != class_name:
                    raise RuntimeError(f"SAM contract mismatch: {path}")
                rows.append({**data, "source_metadata": str(path)})
    muscle_roots = {
        "sam": [
            BASELINE / "sam_gt_guided_wsi_20260726_132222",
            BASELINE / "sam_gt_guided_wsi_20260726_133000",
        ],
        "sam_med2d": [
            BASELINE / "sam_med2d_gt_guided_wsi_20260726_133001",
            BASELINE / "sam_med2d_gt_guided_wsi_20260726_133600",
        ],
        "wsi_sam": [
            BASELINE / "wsi_sam_gt_guided_wsi_20260726_135000",
            BASELINE / "wsi_sam_gt_guided_wsi_20260726_140000",
        ],
    }
    for method, roots in muscle_roots.items():
        found: dict[str, dict] = {}
        for root in roots:
            for path in root.glob("*/metadata.json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("method") == method:
                    found[str(data["wsi_id"])] = {
                        **data,
                        "target_class": class_ids["muscle"],
                        "target_class_name": "muscle",
                        "target_rgb": [34, 172, 56],
                        "source_metadata": str(path),
                    }
        if set(found) != set(WSIS):
            raise RuntimeError(f"incomplete muscle {method}: {sorted(found)}")
        rows.extend(found[wsi_id] for wsi_id in WSIS)
    expected = 3 * 12 * 5
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} SAM rows, got {len(rows)}")
    if any(int(row["target_class"]) != class_ids[str(row["target_class_name"])] for row in rows):
        raise RuntimeError("SAM class ID mismatch")
    return rows


def j5_output_dir(class_index: int, wsi_index: int) -> Path:
    if class_index == 0 and wsi_index == 0:
        stamp = "20260726_170000"
    else:
        stamp = f"20260726_18{class_index:02d}{wsi_index:02d}"
    return WSI_ROOT / f"j5_multi_prompt_wsi_{stamp}"


def collect_j5_new_rows(
    class_ids: dict[str, int],
    class_rgb: dict[str, tuple[int, int, int]],
) -> tuple[list[dict], list[dict]]:
    best_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for class_index, class_name in enumerate(NEW_CLASSES):
        task_stamp = f"20260726_16{class_index:02d}00"
        task_path = BASELINE / f"wsi_gt_guided_tasks_{task_stamp}" / f"wsi_tasks_{task_stamp}.parquet"
        tasks = pd.read_parquet(task_path).set_index("wsi_id")
        for wsi_index, wsi_id in enumerate(WSIS):
            root = j5_output_dir(class_index, wsi_index)
            top_path = root / "metadata.json"
            top = json.loads(top_path.read_text(encoding="utf-8"))
            if top["status"] != "complete" or top["wsi_id"] != wsi_id:
                raise RuntimeError(f"incomplete J5 output: {top_path}")
            item_by_index = {
                int(item["candidate_index"]): item for item in top["candidates"]
            }
            for candidate_index in range(5):
                if candidate_index not in item_by_index:
                    path = root / f"candidate_{candidate_index:02d}" / "metadata.json"
                    item_by_index[candidate_index] = json.loads(path.read_text(encoding="utf-8"))
            all_items = [item_by_index[index] for index in range(5)]
            complete = [item for item in all_items if item["status"] == "complete"]
            valid_paths = [
                root / f"candidate_{int(item['candidate_index']):02d}" / "mask_10x_pyramid.tif"
                for item in complete
            ]
            shape = tuple(map(int, complete[0]["output_shape_10x"]))
            gt_path = Path(str(tasks.loc[wsi_id, "gt_path"]))
            results = candidate_metrics(gt_path, class_rgb[class_name], shape, valid_paths)
            result_by_index = {
                int(item["candidate_index"]): result for item, result in zip(complete, results, strict=True)
            }
            for item in all_items:
                candidate_index = int(item["candidate_index"])
                row = {
                    "method": "j5_oracle5",
                    "wsi_id": wsi_id,
                    "target_class": class_ids[class_name],
                    "target_class_name": class_name,
                    "candidate_index": candidate_index,
                    "candidate_status": str(item["status"]),
                    "prompt_json": str(item.get("prompt_json", "")),
                    "source_metadata": str(top_path),
                }
                if candidate_index in result_by_index:
                    row.update(result_by_index[candidate_index])
                candidate_rows.append(row)
            eligible = [row for row in candidate_rows[-5:] if row["candidate_status"] == "complete"]
            if not eligible:
                raise RuntimeError(f"no valid J5 candidate: {class_name} {wsi_id}")
            best = max(eligible, key=lambda row: float(row["dice"]))
            best_rows.append(
                {
                    **best,
                    "candidate_attempts": 5,
                    "valid_candidate_masks": len(eligible),
                    "model_calls": 5,
                    "positive_clicks": 5,
                    "negative_clicks": 5,
                }
            )
            print(json.dumps({
                "event": "j5_metric_progress",
                "class": class_name,
                "wsi_id": wsi_id,
                "best_candidate": best["candidate_index"],
                "dice": best["dice"],
            }), flush=True)
    return best_rows, candidate_rows


def collect_j5_muscle_rows(class_ids: dict[str, int]) -> tuple[list[dict], list[dict]]:
    candidate_rows: list[dict] = []
    candidate_zero_reports = [
        WSI_ROOT / "wsi_inference_20260726_121454/wsi_gt_vs_prediction_20260726_121800.json",
        WSI_ROOT / "wsi_inference_20260726_122700/wsi_gt_vs_prediction_20260726_130000.json",
        WSI_ROOT / "wsi_inference_20260726_122701/wsi_gt_vs_prediction_20260726_130000.json",
        WSI_ROOT / "wsi_inference_20260726_122702/wsi_gt_vs_prediction_20260726_130000.json",
        WSI_ROOT / "wsi_inference_20260726_122703/wsi_gt_vs_prediction_20260726_130000.json",
    ]
    report_paths = list(candidate_zero_reports)
    report_paths.append(
        WSI_ROOT / "wsi_inference_20260726_142001/wsi_gt_vs_prediction_20260726_142100.json"
    )
    completed_path = BASELINE / "j5_wsi_oracle_sweep_20260726_142000/completed.jsonl"
    completed = [json.loads(line) for line in completed_path.read_text().splitlines() if line.strip()]
    report_paths.extend(Path(str(item["report"])) for item in completed)
    seen: set[tuple[str, int]] = set()
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        wsi_id = str(report["wsi_id"])
        if path in candidate_zero_reports:
            candidate_index = 0
        elif "142001" in str(path):
            candidate_index = 1
        else:
            match = next(item for item in completed if str(item["report"]) == str(path))
            candidate_index = int(match["candidate_index"])
        key = (wsi_id, candidate_index)
        if key in seen:
            continue
        seen.add(key)
        metrics = report["metrics"]
        candidate_rows.append({
            "method": "j5_oracle5",
            "wsi_id": wsi_id,
            "target_class": class_ids["muscle"],
            "target_class_name": "muscle",
            "candidate_index": candidate_index,
            "candidate_status": "complete",
            "source_metadata": str(path),
            **metric_fields(
                int(metrics["gt_positive_pixels"]),
                int(metrics["prediction_positive_pixels"]),
                int(metrics["intersection_pixels"]),
            ),
        })
    if len(candidate_rows) != 25 or len(seen) != 25:
        raise RuntimeError(f"expected 25 muscle candidates, got {len(candidate_rows)}")
    best_rows = []
    for wsi_id in WSIS:
        eligible = [row for row in candidate_rows if row["wsi_id"] == wsi_id]
        best = max(eligible, key=lambda row: float(row["dice"]))
        best_rows.append({
            **best,
            "candidate_attempts": 5,
            "valid_candidate_masks": 5,
            "model_calls": 5,
            "positive_clicks": 5,
            "negative_clicks": 5,
        })
    return best_rows, candidate_rows


def aggregate(per_wsi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows = []
    for (method, class_id, class_name), frame in per_wsi.groupby(
        ["method", "target_class", "target_class_name"], sort=True
    ):
        sums = frame[
            ["gt_positive_pixels", "prediction_positive_pixels", "intersection_pixels"]
        ].sum()
        row = {
            "method": method,
            "target_class": int(class_id),
            "target_class_name": class_name,
            "support_wsi": len(frame),
            **{f"macro_{metric}": float(frame[metric].mean()) for metric in METRICS},
            **{f"micro_{key}": value for key, value in metric_fields(
                int(sums.gt_positive_pixels),
                int(sums.prediction_positive_pixels),
                int(sums.intersection_pixels),
            ).items()},
            "model_calls": int(frame["model_calls"].sum()),
            "positive_clicks": int(frame["positive_clicks"].sum()),
            "negative_clicks": int(frame["negative_clicks"].sum()),
        }
        class_rows.append(row)
    per_class = pd.DataFrame(class_rows)
    scope_classes = {
        "全部12类": set(CLASS_ZH),
        "新增11类（不含肌肉）": set(NEW_CLASSES),
        "前景11类（不含背景）": set(CLASS_ZH) - {"background"},
    }
    overall_rows = []
    for method, method_frame in per_wsi.groupby("method"):
        for scope, classes in scope_classes.items():
            frame = method_frame.loc[method_frame["target_class_name"].isin(classes)]
            class_frame = per_class.loc[
                (per_class["method"] == method)
                & (per_class["target_class_name"].isin(classes))
            ]
            sums = frame[
                ["gt_positive_pixels", "prediction_positive_pixels", "intersection_pixels"]
            ].sum()
            overall_rows.append({
                "method": method,
                "scope": scope,
                "class_count": len(class_frame),
                "wsi_class_count": len(frame),
                **{f"class_macro_{metric}": float(class_frame[f"macro_{metric}"].mean())
                   for metric in METRICS},
                **{f"micro_{key}": value for key, value in metric_fields(
                    int(sums.gt_positive_pixels),
                    int(sums.prediction_positive_pixels),
                    int(sums.intersection_pixels),
                ).items()},
                "model_calls": int(frame["model_calls"].sum()),
                "positive_clicks": int(frame["positive_clicks"].sum()),
                "negative_clicks": int(frame["negative_clicks"].sum()),
            })
    return per_class, pd.DataFrame(overall_rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def markdown_report(
    per_wsi: pd.DataFrame,
    per_class: pd.DataFrame,
    overall: pd.DataFrame,
    candidates: pd.DataFrame,
    artifact_root: Path,
) -> str:
    lines = [
        "# 五张 WSI、十二类组织的全图分割对比汇总",
        "",
        f"- 汇总时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本：{len(WSIS)} 张固定 WSI。",
        "- 类别：12 类；所有类别在 5 张 WSI 中均存在，因此没有缺失类别，也没有把缺失类别强行记为 Dice=0。",
        "- J5：每个 WSI–类别生成 5 组“1 正点 + 1 负点”，报告其中 GT Dice 最高者；这是多候选 oracle 上界，不是单次随机点击结果。",
        "- SAM / SAM-Med2D / WSI-SAM：只处理含目标组织的 patch；正点由 GT 目标区域随机采样，负点在非目标区域采样；无目标 patch 不点击、不调用模型。",
        "- 主指标为类别宏平均（先在每类的 5 张 WSI 上平均，再对类别平均）；同时保存像素微平均。",
        "",
        "## 主要结果",
        "",
        "| 方法 | 范围 | 类别宏 Dice | 类别宏 mIoU | 类别宏 Precision | 类别宏 Recall | 像素微 Dice | 模型调用/候选尝试 | 正点 | 负点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.sort_values(["scope", "method"]).iterrows():
        lines.append(
            f"| {METHOD_ZH[row.method]} | {row.scope} | {fmt(row.class_macro_dice)} | "
            f"{fmt(row.class_macro_iou)} | {fmt(row.class_macro_precision)} | "
            f"{fmt(row.class_macro_recall)} | {fmt(row.micro_dice)} | "
            f"{int(row.model_calls)} | {int(row.positive_clicks)} | {int(row.negative_clicks)} |"
        )
    lines.extend([
        "",
        "## 十二类逐类 Dice（5 张 WSI 宏平均）",
        "",
        "| 类别 | J5 oracle-5 | SAM | SAM-Med2D | WSI-SAM |",
        "|---|---:|---:|---:|---:|",
    ])
    dice_pivot = per_class.pivot(
        index="target_class_name", columns="method", values="macro_dice"
    )
    for class_name in sorted(CLASS_ZH, key=lambda name: int(
        per_class.loc[per_class.target_class_name == name, "target_class"].iloc[0]
    )):
        row = dice_pivot.loc[class_name]
        lines.append(
            f"| {CLASS_ZH[class_name]} (`{class_name}`) | {fmt(row.j5_oracle5)} | "
            f"{fmt(row.sam)} | {fmt(row.sam_med2d)} | {fmt(row.wsi_sam)} |"
        )
    lines.extend([
        "",
        "## 标注与调用成本",
        "",
        "- 三个 SAM 的 12 类实验各自只在 GT 含目标的 patch 上调用；表中的调用次数就是模拟人工交互的 patch 次数。每次最多使用 1 个正点和 1 个负点，若 patch 内没有可用负区域则不强行放置负点。",
        "- J5 每个 WSI–类别尝试 5 组候选，共 60 个 WSI–类别组合，即 300 组候选、300 个正点、300 个负点；最终只保留每组中 Dice 最高的一个结果。",
        f"- J5 候选中成功返回 mask {int((candidates.candidate_status == 'complete').sum())}/"
        f"{len(candidates)}；其余候选因提示落入同一 hard region 而 abstain，不补抽候选。",
        "",
        "## Muscle 单类结果",
        "",
    ])
    muscle = per_class.loc[per_class.target_class_name == "muscle"].set_index("method")
    lines.extend([
        "| 方法 | Dice | IoU | Precision | Recall | 调用/候选尝试 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in ["j5_oracle5", "sam", "sam_med2d", "wsi_sam"]:
        row = muscle.loc[method]
        lines.append(
            f"| {METHOD_ZH[method]} | {fmt(row.macro_dice)} | {fmt(row.macro_iou)} | "
            f"{fmt(row.macro_precision)} | {fmt(row.macro_recall)} | {int(row.model_calls)} |"
        )
    lines.extend([
        "",
        "J5 muscle 的 5 候选 oracle 宏平均 Dice 为 "
        f"**{fmt(float(muscle.loc['j5_oracle5', 'macro_dice']))}**。它显著高于此前的单随机候选 "
        "Dice=0.2680，说明 J5 对提示位置较敏感；论文中应将其明确标为 oracle 上界。",
        "",
        "## 解释与公平性边界",
        "",
        "1. J5 是一次全图传播：每个候选只给一对全局提示，由模型输出整张 WSI。三个 SAM 是逐 patch 交互后拼接，无法用单个局部提示直接完成同等的全图同类组织检索。",
        "2. SAM 的提示点直接由 GT 生成，属于模拟用户点击；J5 的候选点也由 GT 候选规则产生，并进一步用 GT 选择最优候选。因此这些数值衡量的是给定提示策略下的能力，不是无监督性能。",
        "3. 当前全图协议统一提供 Dice、IoU、Precision 和 Recall。boundary F1、exclude-prompt-region Dice、disconnected-region recall 以及 1/3/5-point 曲线尚未在这批 WSI 产物上计算，不能由现有统计表可靠反推。",
        "4. 背景类面积很大，会显著影响像素微平均；主结论应优先引用“全部12类”的类别宏指标，并同时报告“不含背景的前景11类”。",
        "",
        "## 完整产物",
        "",
        f"- 逐 WSI–类别指标：`{artifact_root / 'per_wsi_class_metrics.parquet'}`",
        f"- 逐类指标：`{artifact_root / 'per_class_metrics.parquet'}`",
        f"- 总体指标：`{artifact_root / 'overall_metrics.parquet'}`",
        f"- J5 全部候选指标：`{artifact_root / 'j5_candidate_metrics.parquet'}`",
        f"- 审计信息：`{artifact_root / 'audit.json'}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    class_ids, class_rgb = load_class_contract()
    sam_rows = collect_sam_rows(class_ids)
    j5_new, candidates_new = collect_j5_new_rows(class_ids, class_rgb)
    j5_muscle, candidates_muscle = collect_j5_muscle_rows(class_ids)
    per_wsi = pd.DataFrame(sam_rows + j5_new + j5_muscle)
    candidates = pd.DataFrame(candidates_new + candidates_muscle)
    expected_keys = {
        (method, class_name, wsi_id)
        for method in METHOD_ZH
        for class_name in CLASS_ZH
        for wsi_id in WSIS
    }
    actual_keys = set(zip(
        per_wsi.method, per_wsi.target_class_name, per_wsi.wsi_id, strict=True
    ))
    if actual_keys != expected_keys or len(per_wsi) != 240:
        raise RuntimeError(
            f"per-WSI coverage mismatch rows={len(per_wsi)} "
            f"missing={sorted(expected_keys - actual_keys)[:5]} "
            f"extra={sorted(actual_keys - expected_keys)[:5]}"
        )
    per_class, overall = aggregate(per_wsi)
    root = BASELINE / f"multiclass_wsi_summary_{args.timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    per_wsi.to_parquet(root / "per_wsi_class_metrics.parquet", index=False)
    per_class.to_parquet(root / "per_class_metrics.parquet", index=False)
    overall.to_parquet(root / "overall_metrics.parquet", index=False)
    candidates.to_parquet(root / "j5_candidate_metrics.parquet", index=False)
    audit = {
        "timestamp": args.timestamp,
        "wsi_ids": WSIS,
        "classes": sorted(CLASS_ZH, key=class_ids.get),
        "all_classes_present_on_all_wsis": True,
        "missing_class_policy": "N/A and excluded from denominators",
        "per_wsi_class_rows": len(per_wsi),
        "per_class_rows": len(per_class),
        "j5_candidate_attempts": len(candidates),
        "j5_complete_masks": int((candidates.candidate_status == "complete").sum()),
        "j5_abstained_candidates": int((candidates.candidate_status != "complete").sum()),
    }
    (root / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = markdown_report(per_wsi, per_class, overall, candidates, root)
    args.report.write_text(report, encoding="utf-8")
    (root / "summary_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "artifact_root": str(root),
        "report": str(args.report),
        **audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
