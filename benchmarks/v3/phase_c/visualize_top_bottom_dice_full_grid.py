# 作用：按类别和尺度分别可视化 Phase C Dice 最好/中间/最差案例的旧版 2x4 整图。

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import predict_top_area
from benchmarks.gdph_v2.pret_visualize import (
    CLASS_NAMES,
    PALETTE,
    boundaries_overlay,
    colorize_labels,
    save_single_panel,
    segment_values_to_image,
)


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MULTISCALE_ROOT = PRET_ROOT / "multiscale_tokens"
PROMPT_PATH = PRET_ROOT / "prompt_tasks" / "all_prompt_tasks.csv"
MANIFEST_PATH = PRET_ROOT / "data_manifest_v3.csv"
METRICS_PATH = PRET_ROOT / "evaluations" / "multiscale_baseline_metrics.csv"
SCORE_DIR = PRET_ROOT / "evaluations" / "query_scale_scores"
VIS_ROOT = PRET_ROOT / "visualizations" / "phase_c_dice_class_scale_full_grid"
REPORT_PATH = Path(__file__).resolve().parent / "dice_class_scale_full_grid_report.md"

INDEX_FIELDS = [
    "group",
    "bucket",
    "rank",
    "query_id",
    "image_id",
    "target_class",
    "scale",
    "prompt_quality",
    "prompt_mode",
    "Dice_classwise_toparea",
    "BestDice",
    "mAP",
    "AUROC",
    "path",
    "image",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(tmp, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_ids(value: object) -> list[int]:
    text = str(value or "")
    if not text:
        return []
    return [int(item) for item in text.split(";") if item]


def _resize_arrays(
    rgb: np.ndarray,
    segments: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not max_dim or max(segments.shape) <= max_dim:
        return rgb, segments
    # 直接从 mmap 按目标尺寸取样，避免先复制完整 10x 图像造成多 case 并发时内存峰值过高。
    scale = max_dim / max(segments.shape)
    out_h = max(1, int(round(segments.shape[0] * scale)))
    out_w = max(1, int(round(segments.shape[1] * scale)))
    ys = np.linspace(0, segments.shape[0] - 1, out_h, dtype=np.intp)
    xs = np.linspace(0, segments.shape[1] - 1, out_w, dtype=np.intp)
    rr = np.asarray(rgb[ys[:, None], xs[None, :], :], dtype=np.uint8)
    ss = np.asarray(segments[ys[:, None], xs[None, :]], dtype=np.int32)
    return rr, ss


def resize_raw_gt(gt_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(gt_path)
    size = (target_shape[1], target_shape[0])
    if image.mode in {"1", "L", "I", "I;16"}:
        label = np.asarray(image.resize(size, Image.Resampling.NEAREST))
        return colorize_labels(label.astype(np.int64))
    return np.asarray(image.convert("RGB").resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def gt_labels_from_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    purities = np.asarray([float(row.get("gt_purity", row.get("gt_target_fraction", 0.0))) for row in rows], dtype=np.float32)
    valid = np.asarray([float(row.get("valid_fraction", 0.0)) > 0 for row in rows], dtype=bool)
    return labels, purities, valid


def predicted_mask(metric: dict[str, str], score_npz: np.lib.npyio.NpzFile) -> np.ndarray:
    scores = score_npz["score_final"].astype(np.float32)
    areas = score_npz["area"].astype(np.float64)
    pred_area = float(metric.get("PredArea") or 0.0)
    fraction = pred_area / max(float(np.sum(areas)), 1.0)
    return predict_top_area(scores, areas, float(np.clip(fraction, 0.001, 0.95)))


def cell_density(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row.get("cell_density", 0.0)) for row in rows], dtype=np.float32)


def pseudo_patch_id(rows: list[dict[str, str]]) -> np.ndarray:
    centers = np.asarray([[float(row.get("center_x", 0.0)), float(row.get("center_y", 0.0))] for row in rows], dtype=np.float32)
    if centers.size == 0:
        return np.zeros(0, dtype=np.float32)
    grid_x = np.floor(centers[:, 0] / 1024.0)
    grid_y = np.floor(centers[:, 1] / 1024.0)
    return (grid_y * (float(grid_x.max()) + 1.0) + grid_x).astype(np.float32)


def label_image_from_segments(segments: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe_labels = np.full(len(labels), 255, dtype=np.uint8)
    keep = valid & (labels >= 0) & (labels < len(PALETTE))
    safe_labels[keep] = labels[keep].astype(np.uint8)
    out = np.full(segments.shape, 255, dtype=np.uint8)
    mask = segments >= 0
    out[mask] = safe_labels[np.maximum(segments[mask], 0)]
    return out


def combine_case_panels(case_dir: Path) -> Path:
    names = [
        "01_he_superpixel_boundary.png",
        "02_prompt_overlay.png",
        "03_gt_tissue_mask.png",
        "04_score_heatmap.png",
        "05_predicted_mask.png",
        "06_error_map.png",
        "07_cell_density_heatmap.png",
        "08_patch_id_map.png",
    ]
    panels = [Image.open(case_dir / name).convert("RGB") for name in names]
    target_width = 760
    resized = []
    for panel in panels:
        scale = target_width / panel.width
        resized.append(panel.resize((target_width, max(1, int(round(panel.height * scale)))), Image.Resampling.BILINEAR))
    cols = 4
    row_heights = [max(image.height for image in resized[row * cols:(row + 1) * cols]) for row in range(2)]
    canvas = Image.new("RGB", (target_width * cols, sum(row_heights)), "white")
    y = 0
    for row in range(2):
        x = 0
        for image in resized[row * cols:(row + 1) * cols]:
            canvas.paste(image, (x, y))
            x += target_width
        y += row_heights[row]
    output = case_dir / "00_combined_grid.png"
    canvas.save(output)
    for panel in panels:
        panel.close()
    for image in resized:
        image.close()
    canvas.close()
    return output


def remove_individual_panels(case_dir: Path) -> None:
    for path in case_dir.glob("[0-9][0-9]_*.png"):
        if path.name != "00_combined_grid.png":
            path.unlink()


def render_case(
    metric: dict[str, str],
    prompt: dict[str, str],
    manifest: dict[str, str],
    vis_root: Path,
    group: str,
    bucket: str,
    rank: int,
    max_render_dimension: int,
    keep_individual_panels: bool,
    skip_existing: bool,
) -> tuple[Path, Path]:
    image_id = metric["image_id"]
    scale = metric["scale"]
    class_id = int(metric["target_class"])
    scale_dir = MULTISCALE_ROOT / image_id / scale
    case_dir = (
        vis_root
        / "visual_summary"
        / f"class_{class_id:02d}"
        / scale
        / group
        / f"{safe_name(metric['query_id'])}_{scale}_rank{rank:03d}_dice{float(metric['Dice_classwise_toparea']):.3f}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    combined_path = case_dir / "00_combined_grid.png"
    if skip_existing and combined_path.exists():
        return case_dir, combined_path

    rows = read_csv(scale_dir / "superpixels.csv")
    labels, purities, valid = gt_labels_from_rows(rows)
    rgb = np.asarray(np.load(scale_dir / "he_10x_rgb.npy", mmap_mode="r"))
    segments = np.asarray(np.load(scale_dir / "superpixels.npy", mmap_mode="r"))
    score_npz = np.load(SCORE_DIR / f"{metric['query_id']}_{scale}.npz")
    scores = score_npz["score_final"].astype(np.float32)
    pred = predicted_mask(metric, score_npz)
    target = labels == class_id

    render_rgb, render_segments = _resize_arrays(rgb, segments, max_render_dimension)
    render_gt = resize_raw_gt(Path(manifest["gt_mask_path"]), render_segments.shape)

    positive_ids = score_npz["positive_prompt_segments"].astype(np.int64).tolist()
    negative_ids = score_npz["negative_prompt_segments"].astype(np.int64).tolist()
    prompt_mask = np.zeros(len(rows), dtype=bool)
    negative_mask = np.zeros(len(rows), dtype=bool)
    if positive_ids:
        prompt_mask[np.asarray(positive_ids, dtype=np.int64)] = True
    if negative_ids:
        negative_mask[np.asarray(negative_ids, dtype=np.int64)] = True

    prompt_colors = np.full((len(rows), 3), [230, 230, 230], dtype=np.uint8)
    prompt_colors[prompt_mask] = [0, 80, 255]
    prompt_colors[negative_mask] = [230, 0, 140]
    pred_colors = np.full((len(rows), 3), [230, 230, 230], dtype=np.uint8)
    pred_colors[pred] = [20, 180, 70]
    error_colors = np.full((len(rows), 3), [230, 230, 230], dtype=np.uint8)
    error_colors[pred & target] = [20, 170, 60]
    error_colors[pred & ~target] = [230, 0, 140]
    error_colors[~pred & target & valid] = [90, 90, 90]

    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
    common = [
        f"query={metric['query_id']} | image={image_id} | scale={scale}",
        f"class={class_id} {class_name} | quality={metric.get('prompt_quality', '')} | mode={metric.get('prompt_mode', '')}",
        f"Dice={float(metric['Dice_classwise_toparea']):.3f} | BestDice={float(metric['BestDice']):.3f} | mAP={float(metric['mAP']):.3f} | AUROC={float(metric['AUROC']):.3f}",
        f"Precision={float(metric['Precision']):.3f} | Recall={float(metric['Recall']):.3f} | pos={len(positive_ids)} neg={len(negative_ids)}",
    ]

    save_single_panel(
        case_dir / "01_he_superpixel_boundary.png",
        "01 HE + superpixel boundary",
        boundaries_overlay(render_rgb, render_segments),
        common,
    )
    save_single_panel(
        case_dir / "02_prompt_overlay.png",
        "02 Prompt overlay",
        segment_values_to_image(render_segments, np.zeros(len(rows)), prompt_colors),
        common + ["blue=positive prompt; magenta=negative prompt; gray=unused"],
    )
    save_single_panel(
        case_dir / "03_gt_tissue_mask.png",
        "03 Raw GT tissue mask",
        render_gt,
        common + ["raw annotation from data_manifest_v3 gt_mask_path"],
    )
    save_single_panel(
        case_dir / "04_score_heatmap.png",
        "04 Similarity score heatmap",
        segment_values_to_image(render_segments, scores),
        common + ["blue=low score; green=mid; red=high score"],
        add_heat_legend=True,
    )
    save_single_panel(
        case_dir / "05_predicted_mask.png",
        "05 Predicted target mask",
        segment_values_to_image(render_segments, np.zeros(len(rows)), pred_colors),
        common + ["green=predicted target; gray=not predicted"],
    )
    save_single_panel(
        case_dir / "06_error_map.png",
        "06 Error map",
        segment_values_to_image(render_segments, np.zeros(len(rows)), error_colors),
        common + ["green=TP; magenta=FP; gray=FN; light gray=TN/ignored"],
    )
    save_single_panel(
        case_dir / "07_cell_density_heatmap.png",
        "07 Cell density heatmap",
        segment_values_to_image(render_segments, cell_density(rows)),
        common + ["blue=low; green=mid; red=high"],
        add_heat_legend=True,
    )
    save_single_panel(
        case_dir / "08_patch_id_map.png",
        "08 1024-grid pseudo patch id map",
        segment_values_to_image(render_segments, pseudo_patch_id(rows)),
        common + ["diagnostic only; colors encode 1024-grid location"],
        add_heat_legend=True,
    )
    legend = [
        f"group: {group}",
        f"bucket: {bucket}",
        f"rank: {rank}",
        f"query_id: {metric['query_id']}",
        f"image_id: {image_id}",
        f"scale: {scale}",
        f"class: {class_id} {class_name}",
        f"Dice_classwise_toparea: {metric['Dice_classwise_toparea']}",
        f"BestDice: {metric['BestDice']}",
        f"mAP: {metric['mAP']}",
        f"AUROC: {metric['AUROC']}",
        "",
        "Raw GT tissue mask: original annotation image from manifest.",
        "Prompt overlay: blue positive, magenta negative.",
        "Error map: green TP, magenta FP, gray FN, light gray TN/ignored.",
        "Error map and metrics use superpixel gt_majority_label to match Phase C evaluation.",
        "Score heatmap: blue low, green middle, red high.",
        "",
        "GDPH palette:",
    ]
    legend.extend(f"{idx}: {name} rgb={PALETTE[idx].tolist()}" for idx, name in enumerate(CLASS_NAMES))
    (case_dir / "00_legend.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")
    summary = {**metric, "group": group, "bucket": bucket, "rank": rank, "combined_grid": "00_combined_grid.png"}
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    combined = combine_case_panels(case_dir)
    if not keep_individual_panels:
        remove_individual_panels(case_dir)
    return case_dir, combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5, help="cases per class-scale bucket")
    parser.add_argument("--max_render_dimension", type=int, default=4096)
    parser.add_argument("--keep_individual_panels", action="store_true")
    parser.add_argument("--vis_root", type=Path, default=VIS_ROOT)
    parser.add_argument(
        "--buckets",
        nargs="+",
        choices=("worst", "middle", "best"),
        default=("worst", "middle", "best"),
        help="要生成的 Dice 分桶，默认三者全部生成",
    )
    parser.add_argument("--class_ids", nargs="+", type=int, help="仅渲染指定类别 id")
    parser.add_argument("--scales", nargs="+", choices=("small", "medium", "large"), help="仅渲染指定尺度")
    parser.add_argument("--workers", type=int, default=1, help="case 级并发渲染数")
    parser.add_argument("--skip_existing", action="store_true", help="已有完整合图时直接复用，不重复渲染")
    return parser.parse_args()


def select_middle(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return list(rows)
    center = len(rows) // 2
    start = max(0, center - count // 2)
    end = start + count
    if end > len(rows):
        end = len(rows)
        start = max(0, end - count)
    return rows[start:end]


def select_class_scale_buckets(metrics: list[dict[str, str]], count: int) -> list[tuple[str, str, list[dict[str, str]]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        grouped[(int(row["target_class"]), row["scale"])].append(row)
    selected: list[tuple[str, str, list[dict[str, str]]]] = []
    for class_id, scale in sorted(grouped, key=lambda item: (item[0], {"small": 0, "medium": 1, "large": 2}.get(item[1], 99))):
        rows = sorted(grouped[(class_id, scale)], key=lambda row: float(row["Dice_classwise_toparea"]))
        prefix = f"class_{class_id:02d}_{scale}"
        selected.append((f"{prefix}_worst", "worst", rows[:count]))
        selected.append((f"{prefix}_middle", "middle", select_middle(rows, count)))
        selected.append((f"{prefix}_best", "best", list(reversed(rows[-count:]))))
    return selected


def main() -> None:
    args = parse_args()
    metrics = [row for row in read_csv(METRICS_PATH) if row.get("status") == "ok"]
    prompts = {row["query_id"]: row for row in read_csv(PROMPT_PATH)}
    manifests = {row["image_id"]: row for row in read_csv(MANIFEST_PATH)}
    index_rows: list[dict[str, object]] = []
    requested_buckets = set(args.buckets)
    selections = [
        selection
        for selection in select_class_scale_buckets(metrics, args.count)
        if selection[1] in requested_buckets
    ]
    if args.class_ids is not None:
        requested_class_ids = set(args.class_ids)
        selections = [
            selection for selection in selections if int(selection[0].split("_")[1]) in requested_class_ids
        ]
    if args.scales is not None:
        requested_scales = set(args.scales)
        selections = [selection for selection in selections if selection[0].split("_")[2] in requested_scales]
    jobs = [
        (group, bucket, rank, metric)
        for group, bucket, selected in selections
        for rank, metric in enumerate(selected, start=1)
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                render_case,
                metric,
                prompts[metric["query_id"]],
                manifests[metric["image_id"]],
                args.vis_root,
                group,
                bucket,
                rank,
                args.max_render_dimension,
                args.keep_individual_panels,
                args.skip_existing,
            ): (group, bucket, rank, metric)
            for group, bucket, rank, metric in jobs
        }
        for future in as_completed(futures):
            group, bucket, rank, metric = futures[future]
            case_dir, combined = future.result()
            index_rows.append(
                {
                    "group": group,
                    "bucket": bucket,
                    "rank": rank,
                    "query_id": metric["query_id"],
                    "image_id": metric["image_id"],
                    "target_class": metric["target_class"],
                    "scale": metric["scale"],
                    "prompt_quality": metric["prompt_quality"],
                    "prompt_mode": metric["prompt_mode"],
                    "Dice_classwise_toparea": metric["Dice_classwise_toparea"],
                    "BestDice": metric["BestDice"],
                    "mAP": metric["mAP"],
                    "AUROC": metric["AUROC"],
                    "path": str(case_dir),
                    "image": str(combined),
                }
            )
            print(f"full_grid {group} rank={rank} query={metric['query_id']} scale={metric['scale']}", flush=True)
    index_path = args.vis_root / "visual_summary_index.csv"
    write_csv(index_path, index_rows, INDEX_FIELDS)
    summary = {
        "metric": "Dice_classwise_toparea",
        "buckets": sorted(requested_buckets),
        "count_per_class_scale_bucket": args.count,
        "group_count": len(selections),
        "rendered_cases": len(index_rows),
        "bucket_counts": dict(Counter(row["bucket"] for row in index_rows)),
        "index": str(index_path),
        "output": str(args.vis_root / "visual_summary"),
        "gt_panel": "raw annotation from data_manifest_v3 gt_mask_path",
        "error_map_target": "superpixel gt_majority_label, matching Phase C metrics",
    }
    (args.vis_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Phase C Dice Top/Bottom Full-grid Visualization",
                "",
                f"- output: {summary['output']}",
                f"- index: {summary['index']}",
                f"- metric: {summary['metric']}",
                f"- count per class-scale bucket: {summary['count_per_class_scale_bucket']}",
                f"- rendered cases: {summary['rendered_cases']}",
                f"- bucket counts: `{summary['bucket_counts']}`",
                f"- GT panel: {summary['gt_panel']}",
                f"- error map target: {summary['error_map_target']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
