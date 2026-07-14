from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.gdph_v2.pret_aaai_eval import CLASS_NAMES
from benchmarks.gdph_v2.pret_aaai_next_experiments import (
    _interaction_groups,
    _prompt_group_stats,
    _score_with_prompts,
)
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    otsu_score_threshold,
    percentile_threshold,
    prompt_derived_class,
    predict_top_area,
    read_csv,
    segment_adjacency,
)
from benchmarks.gdph_v2.pret_visualize import (
    PALETTE,
    boundaries_overlay,
    colorize_labels,
    save_single_panel,
    segment_values_to_image,
)


DEFAULT_NEXT_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_aaai_next_full10x"
DEFAULT_SOURCE_ROOT = "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_prompts(rows: list[dict[str, str]], prompt_source: str = "realistic_box") -> list[dict[str, str]]:
    seen: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("prompt_source") != prompt_source:
            continue
        seen.setdefault((row["query_id"], row["prompt_source"], int(row["shot"])), row)
    return [seen[key] for key in sorted(seen)]


def _resize_arrays(rgb: np.ndarray, segments: np.ndarray, gt: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not max_dim or max(segments.shape) <= max_dim:
        return rgb, segments, gt
    scale = max_dim / max(segments.shape)
    size = (max(1, int(round(segments.shape[1] * scale))), max(1, int(round(segments.shape[0] * scale))))
    rr = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize(size, Image.BILINEAR))
    ss = np.asarray(Image.fromarray(segments.astype(np.int32), mode="I").resize(size, Image.NEAREST), dtype=np.int32)
    gg = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize(size, Image.NEAREST), dtype=np.uint8)
    return rr, ss, gg


def _prediction(scores: np.ndarray, areas: np.ndarray, metric: dict[str, str]) -> np.ndarray:
    protocol = metric["threshold_protocol"]
    if protocol == "p90":
        return scores >= percentile_threshold(scores, 90.0)
    if protocol == "p80":
        return scores >= percentile_threshold(scores, 80.0)
    if protocol == "otsu":
        return scores >= otsu_score_threshold(scores)
    if protocol in {"global_toparea_loio", "classwise_toparea_loio", "prompt_adaptive_area_loio", "gmm_2comp"}:
        return predict_top_area(scores, areas, max(0.01, min(0.60, _safe_float(metric.get("pred_area_fraction"), 0.20))))
    return predict_top_area(scores, areas, max(0.01, min(0.60, _safe_float(metric.get("pred_area_fraction"), 0.20))))


def _case_slug(metric: dict[str, str], category: str) -> str:
    shot = metric.get("shot", "")
    shot_suffix = f"_shot{int(float(shot))}" if shot not in {"", None} else ""
    return (
        f"class_{int(metric['class_id']):02d}/"
        f"{category}/"
        f"{metric['query_id']}{shot_suffix}_{metric['interaction_protocol']}_{metric['threshold_protocol']}"
    )


def _combine_case_panels(case_dir: Path) -> Path | None:
    panel_names = [
        "01_he_superpixel_boundary.png",
        "02_prompt_overlay.png",
        "03_gt_tissue_mask.png",
        "04_score_heatmap.png",
        "05_predicted_mask.png",
        "06_error_map.png",
        "07_cell_density_heatmap.png",
        "08_patch_id_map.png",
    ]
    paths = [case_dir / name for name in panel_names]
    if not all(path.exists() for path in paths):
        return None
    panels = [Image.open(path).convert("RGB") for path in paths]
    target_width = 760
    resized = []
    for panel in panels:
        scale = target_width / panel.width
        resized.append(panel.resize((target_width, max(1, int(round(panel.height * scale)))), Image.BILINEAR))
    cols = 4
    rows = 2
    row_heights = [
        max(image.height for image in resized[row * cols:(row + 1) * cols])
        for row in range(rows)
    ]
    canvas = Image.new("RGB", (target_width * cols, sum(row_heights)), "white")
    y = 0
    for row in range(rows):
        x = 0
        for image in resized[row * cols:(row + 1) * cols]:
            canvas.paste(image, (x, y))
            x += target_width
        y += row_heights[row]
    output = case_dir / "00_combined_grid.png"
    canvas.save(output)
    return output


def _individual_panel_paths(case_dir: Path) -> list[Path]:
    return [
        case_dir / "01_he_superpixel_boundary.png",
        case_dir / "02_prompt_overlay.png",
        case_dir / "03_gt_tissue_mask.png",
        case_dir / "04_score_heatmap.png",
        case_dir / "05_predicted_mask.png",
        case_dir / "06_error_map.png",
        case_dir / "07_cell_density_heatmap.png",
        case_dir / "08_patch_id_map.png",
    ]


def _remove_individual_panels(case_dir: Path) -> None:
    for path in _individual_panel_paths(case_dir):
        if path.exists():
            path.unlink()


def _draw_prompt_boxes_on_gt(
    render_gt: np.ndarray,
    render_segments: np.ndarray,
    positive_groups: list[list[int]],
) -> Image.Image:
    image = Image.fromarray(colorize_labels(render_gt)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for group in positive_groups:
        ids = [int(segment_id) for segment_id in group]
        if not ids:
            continue
        mask = np.isin(render_segments, np.asarray(ids, dtype=np.int32))
        if not np.any(mask):
            continue
        yy, xx = np.nonzero(mask)
        x0, x1 = int(xx.min()), int(xx.max())
        y0, y1 = int(yy.min()), int(yy.max())
        width = max(2, int(round(max(image.size) / 512)))
        for offset in range(width):
            draw.rectangle(
                [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                outline=(0, 0, 0),
            )
    return image


def _true_prompt_mask_image(render_segments: np.ndarray, prompt_mask: np.ndarray) -> Image.Image:
    colors = np.full((len(prompt_mask), 3), [28, 28, 28], dtype=np.uint8)
    colors[prompt_mask] = [0, 120, 255]
    return Image.fromarray(segment_values_to_image(render_segments, np.zeros(len(prompt_mask)), colors)).convert("RGB")


def _render_case(
    source_root: Path,
    output_root: Path,
    metric: dict[str, str],
    prompts: list[dict[str, str]],
    all_prompts: list[dict[str, str]],
    category: str,
    lambda_neg: float,
    max_render_dimension: int,
    keep_individual_panels: bool,
    render_mode: str,
    image_cache: dict[str, dict[str, object]] | None = None,
) -> Path:
    image_id = metric["image_id"]
    class_id = int(metric["class_id"])
    query_class_id = int(metric.get("query_class_id", metric.get("class_id", 255)))
    query_id = metric["query_id"]
    if image_cache is not None and image_id in image_cache:
        cached = image_cache[image_id]
        records = cached["records"]
        labels = cached["labels"]
        purities = cached["purities"]
        valid = cached["valid"]
        areas = cached["areas"]
        rgb = cached["rgb"]
        segments = cached["segments"]
        gt = cached["gt"]
        tokens = cached["tokens"]
    else:
        image_dir = source_root / PRET_DIR / image_id
        records = read_csv(image_dir / "superpixels.csv")
        labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
        purities = np.asarray([float(row["gt_label_purity"]) for row in records], dtype=np.float64)
        valid = np.asarray([row["valid_for_retrieval"].lower() == "true" for row in records], dtype=bool)
        areas = np.asarray([float(row["area_10x_pixels"]) for row in records], dtype=np.float64)
        rgb = np.asarray(np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r"))
        segments = np.asarray(np.load(image_dir / "superpixels.npy", mmap_mode="r"))
        gt = np.asarray(np.load(source_root / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r"))
        if gt.shape != segments.shape:
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((segments.shape[1], segments.shape[0]), Image.NEAREST))
        tokens = np.asarray(np.load(image_dir / "tokens_image_cell_reg_cellw0p5.npy", mmap_mode="r"), dtype=np.float32)
        if image_cache is not None:
            image_cache[image_id] = {
                "records": records,
                "labels": labels,
                "purities": purities,
                "valid": valid,
                "areas": areas,
                "rgb": rgb,
                "segments": segments,
                "gt": gt,
                "tokens": tokens,
            }
    metric_shot = metric.get("shot", "")
    if metric_shot not in {"", None}:
        prompt = next(
            row for row in prompts
            if row["query_id"] == query_id and int(row.get("shot", 0)) == int(float(metric_shot))
        )
    else:
        prompt = next(row for row in prompts if row["query_id"] == query_id)

    by_image_class: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_prompts:
        by_image[row["image_id"]].append(row)
        by_image_class[(row["image_id"], int(row["class_id"]))].append(row)
    positive_groups, negative_groups, status, negative_source = _interaction_groups(
        prompt,
        by_image_class,
        by_image,
        tokens,
        labels,
        valid,
        purities,
        metric["interaction_protocol"],
    )
    if status != "ok":
        raise RuntimeError(f"cannot rebuild prompt groups for {query_id}: {status}")
    recomputed_class_id, eval_prompt_class_fraction = prompt_derived_class(
        positive_groups, labels, areas, default=class_id
    )
    positive_stats = _prompt_group_stats(positive_groups, labels, purities, valid, class_id)
    negative_stats = _prompt_group_stats(negative_groups, labels, purities, valid, class_id)
    scores, pos_count, neg_count, compactness = _score_with_prompts(tokens, positive_groups, negative_groups, lambda_neg)
    prompt_mask = np.zeros(len(records), dtype=bool)
    negative_mask = np.zeros(len(records), dtype=bool)
    for group in positive_groups:
        prompt_mask[np.asarray(group, dtype=np.int64)] = True
    for group in negative_groups:
        negative_mask[np.asarray(group, dtype=np.int64)] = True
    candidate = valid & ~prompt_mask if metric["scope"] == "exclude_prompt_region" else valid
    candidate_scores = scores[candidate]
    candidate_areas = areas[candidate]
    pred_candidate = _prediction(candidate_scores, candidate_areas, metric)
    pred = np.zeros(len(records), dtype=bool)
    pred[np.flatnonzero(candidate)[pred_candidate]] = True
    target = labels == class_id

    render_rgb, render_segments, render_gt = _resize_arrays(rgb, segments, gt, max_render_dimension)
    pred_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    pred_colors[pred] = [20, 180, 70]
    prompt_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    prompt_colors[prompt_mask] = [0, 80, 255]
    prompt_colors[negative_mask] = [230, 0, 140]
    error_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    error_colors[pred & target] = [20, 170, 60]
    error_colors[pred & ~target] = [230, 0, 140]
    error_colors[~pred & target & valid] = [90, 90, 90]

    cell_density = np.asarray([float(row["cell_density"]) for row in records], dtype=np.float32)
    patch_ids = np.asarray([float(row["patch_index"]) for row in records], dtype=np.float32)
    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
    query_class_name = CLASS_NAMES[query_class_id] if 0 <= query_class_id < len(CLASS_NAMES) else str(query_class_id)
    recomputed_class_name = CLASS_NAMES[recomputed_class_id] if 0 <= recomputed_class_id < len(CLASS_NAMES) else str(recomputed_class_id)
    case_dir = output_root / PRET_DIR / "visual_summary" / _case_slug(metric, category)
    common = [
        f"query={query_id}",
        f"eval class={class_id} {class_name} | query class={query_class_id} {query_class_name} | protocol={metric['interaction_protocol']} | threshold={metric['threshold_protocol']}",
        f"AP={_safe_float(metric.get('average_precision')):.3f} | Dice={_safe_float(metric.get('dice')):.3f} | pos={pos_count} neg={neg_count} source={negative_source}",
        f"prompt-derived recompute={recomputed_class_id} {recomputed_class_name} fraction={eval_prompt_class_fraction:.3f} | query/eval match={int(query_class_id == class_id)}",
        f"pos GT label={positive_stats['majority_label_mode']} target_frac={positive_stats['target_fraction']:.3f} purity={positive_stats['mean_purity']:.3f}",
        f"neg GT label={negative_stats['majority_label_mode']} target_frac={negative_stats['target_fraction']:.3f} purity={negative_stats['mean_purity']:.3f}",
    ]
    case_dir.mkdir(parents=True, exist_ok=True)
    true_prompt_mask = _true_prompt_mask_image(render_segments, prompt_mask)
    true_prompt_mask.save(case_dir / "true_prompt_mask.png")
    if render_mode == "gt_prompt":
        gt_prompt = _draw_prompt_boxes_on_gt(render_gt, render_segments, positive_groups)
        gt_prompt.save(case_dir / "gt_prompt_box.png")
        summary = dict(metric)
        summary.update(
            {
                "positive_gt_majority_label_recomputed": positive_stats["majority_label_mode"],
                "positive_gt_target_fraction_recomputed": positive_stats["target_fraction"],
                "positive_gt_mean_purity_recomputed": positive_stats["mean_purity"],
                "prompt_derived_class_recomputed": recomputed_class_id,
                "prompt_derived_class_fraction_recomputed": eval_prompt_class_fraction,
                "query_class_id_recomputed": query_class_id,
                "query_eval_class_match_recomputed": int(query_class_id == class_id),
                "render_mode": render_mode,
                "image_file": "true_prompt_mask.png",
                "box_image_file": "gt_prompt_box.png",
                "true_prompt_mask_file": "true_prompt_mask.png",
            }
        )
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return case_dir

    save_single_panel(case_dir / "01_he_superpixel_boundary.png", "01 HE + superpixel boundary", boundaries_overlay(render_rgb, render_segments), common)
    save_single_panel(case_dir / "02_prompt_overlay.png", "02 Prompt overlay", segment_values_to_image(render_segments, np.zeros(len(records)), prompt_colors), common + ["blue=positive prompt; magenta=hard negative prompt; gray=unused"])
    save_single_panel(case_dir / "03_gt_tissue_mask.png", "03 GT tissue mask", colorize_labels(render_gt), common + ["GDPH class palette; see 00_legend.txt"])
    save_single_panel(case_dir / "04_score_heatmap.png", "04 Similarity score heatmap", segment_values_to_image(render_segments, scores), common + ["blue=low score; green=mid; red=high retrieved similarity"], add_heat_legend=True)
    save_single_panel(case_dir / "05_predicted_mask.png", "05 Predicted target mask", segment_values_to_image(render_segments, np.zeros(len(records)), pred_colors), common + ["green=predicted target; gray=not predicted"])
    save_single_panel(case_dir / "06_error_map.png", "06 Error map", segment_values_to_image(render_segments, np.zeros(len(records)), error_colors), common + ["green=TP; magenta=FP; gray=FN; light gray=TN/ignored"])
    save_single_panel(case_dir / "07_cell_density_heatmap.png", "07 Cell density heatmap", segment_values_to_image(render_segments, cell_density), common + ["blue=low; green=mid; red=high"], add_heat_legend=True)
    save_single_panel(case_dir / "08_patch_id_map.png", "08 Nearest patch id map", segment_values_to_image(render_segments, patch_ids), common + ["diagnostic only; colors encode nearest patch id"], add_heat_legend=True)
    legend = [
        f"category: {category}",
        f"query_id: {query_id}",
        f"image_id: {image_id}",
        f"eval_class: {class_id} {class_name}",
        f"query_class: {query_class_id} {query_class_name}",
        f"prompt_derived_recomputed_class: {recomputed_class_id} {recomputed_class_name}",
        f"prompt_derived_recomputed_fraction: {eval_prompt_class_fraction:.6f}",
        f"query_eval_class_match: {int(query_class_id == class_id)}",
        f"interaction_protocol: {metric['interaction_protocol']}",
        f"threshold_protocol: {metric['threshold_protocol']}",
        f"negative_source: {negative_source}",
        f"prototype_compactness: {compactness:.6f}",
        f"positive_gt_majority_label: {positive_stats['majority_label_mode']}",
        f"positive_gt_target_fraction: {positive_stats['target_fraction']:.6f}",
        f"positive_gt_mean_purity: {positive_stats['mean_purity']:.6f}",
        f"negative_gt_majority_label: {negative_stats['majority_label_mode']}",
        f"negative_gt_target_fraction: {negative_stats['target_fraction']:.6f}",
        f"negative_gt_mean_purity: {negative_stats['mean_purity']:.6f}",
        f"negative_gt_min_purity: {negative_stats['min_purity']:.6f}",
        "",
        "Score heatmap: blue low, green middle, red high.",
        "Error map: green TP, magenta FP, gray FN, light gray TN/ignored.",
        "Prompt overlay: blue positive, magenta negative.",
        "",
        "GDPH palette:",
    ]
    legend.extend(f"{idx}: {name} rgb={PALETTE[idx].tolist()}" for idx, name in enumerate(CLASS_NAMES))
    (case_dir / "00_legend.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")
    summary = dict(metric)
    summary.update(
        {
            "positive_gt_majority_label_recomputed": positive_stats["majority_label_mode"],
            "positive_gt_target_fraction_recomputed": positive_stats["target_fraction"],
            "positive_gt_mean_purity_recomputed": positive_stats["mean_purity"],
            "prompt_derived_class_recomputed": recomputed_class_id,
            "prompt_derived_class_fraction_recomputed": eval_prompt_class_fraction,
            "query_class_id_recomputed": query_class_id,
            "query_eval_class_match_recomputed": int(query_class_id == class_id),
            "negative_gt_majority_label_recomputed": negative_stats["majority_label_mode"],
            "negative_gt_target_fraction_recomputed": negative_stats["target_fraction"],
            "negative_gt_mean_purity_recomputed": negative_stats["mean_purity"],
            "negative_gt_min_purity_recomputed": negative_stats["min_purity"],
        }
    )
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _combine_case_panels(case_dir)
    if not keep_individual_panels:
        _remove_individual_panels(case_dir)
    return case_dir


def _pick_cases(rows: list[dict[str, str]], class_id: str, protocol: str, threshold: str, source: str, count: int) -> list[tuple[str, dict[str, str]]]:
    subset = [
        row for row in rows
        if row.get("class_id") == class_id
        and row.get("interaction_protocol") == protocol
        and row.get("threshold_protocol") == threshold
        and row.get("negative_source") == source
        and row.get("scope") == "exclude_prompt_region"
    ]
    subset.sort(key=lambda row: _safe_float(row.get("dice")), reverse=True)
    if not subset:
        return []
    picks: list[tuple[str, dict[str, str]]] = []
    picks.extend(("best", row) for row in subset[:count])
    mid = len(subset) // 2
    picks.extend(("median", row) for row in subset[max(0, mid - count // 2): mid + max(1, count - count // 2)])
    picks.extend(("worst", row) for row in subset[-count:])
    return picks


def _pick_corrections(rows: list[dict[str, str]], class_id: str, count: int) -> list[tuple[str, dict[str, str]]]:
    base = {
        row["query_id"]: row for row in rows
        if row.get("class_id") == class_id
        and row.get("interaction_protocol") == "1pos"
        and row.get("threshold_protocol") == "prompt_adaptive_area_loio"
        and row.get("scope") == "exclude_prompt_region"
    }
    neg = [
        row for row in rows
        if row.get("class_id") == class_id
        and row.get("interaction_protocol") in {"1pos_3strictneg", "1pos_3neg"}
        and row.get("threshold_protocol") == "classwise_toparea_loio"
        and row.get("negative_source") in {"realistic_hard_strict", "realistic_hard"}
        and row.get("scope") == "exclude_prompt_region"
        and row["query_id"] in base
    ]
    neg.sort(key=lambda row: 0 if row.get("negative_source") == "realistic_hard_strict" else 1)
    preferred = [row for row in neg if row.get("negative_source") == "realistic_hard_strict"] or neg
    neg = preferred
    neg.sort(key=lambda row: _safe_float(row.get("dice")) - _safe_float(base[row["query_id"]].get("dice")), reverse=True)
    return [("hard_negative_correction", row) for row in neg[:count]]


def _pick_bottom_half_balanced(
    rows: list[dict[str, str]],
    classes: list[str],
    protocol: str,
    threshold: str,
    source: str,
    max_cases: int,
) -> list[tuple[str, dict[str, str]]]:
    by_class: dict[str, list[dict[str, str]]] = {}
    for class_id in classes:
        subset = [
            row for row in rows
            if row.get("class_id") == class_id
            and row.get("interaction_protocol") == protocol
            and row.get("threshold_protocol") == threshold
            and row.get("negative_source") == source
            and row.get("scope") == "exclude_prompt_region"
        ]
        subset.sort(key=lambda row: _safe_float(row.get("dice")))
        keep = max(1, len(subset) // 2) if subset else 0
        by_class[class_id] = subset[:keep]
    selected: list[tuple[str, dict[str, str]]] = []
    while any(by_class.values()):
        for class_id in classes:
            if by_class[class_id]:
                selected.append(("bottom_half", by_class[class_id].pop(0)))
                if max_cases > 0 and len(selected) >= max_cases:
                    return selected
    return selected


def _existing_case_dir(
    output_root: Path,
    metric: dict[str, str],
    category: str,
    render_mode: str,
) -> Path | None:
    case_dir = output_root / PRET_DIR / "visual_summary" / _case_slug(metric, category)
    image_file = "true_prompt_mask.png" if render_mode == "gt_prompt" else "00_combined_grid.png"
    if (case_dir / image_file).is_file() and (case_dir / "summary.json").is_file():
        return case_dir
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AAAI visual summary cases.")
    parser.add_argument("--next_root", default=DEFAULT_NEXT_ROOT)
    parser.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--prompt_source", default="realistic_box")
    parser.add_argument(
        "--visual_output_root",
        default=None,
        help="Optional output root for visualization files. Metrics are still read from --next_root.",
    )
    parser.add_argument("--render_mode", choices=["full", "gt_prompt"], default="full")
    parser.add_argument("--limit_per_class", type=int, default=1)
    parser.add_argument("--selection_mode", choices=["standard", "bottom_half"], default="standard")
    parser.add_argument("--bottom_protocol", default="1pos")
    parser.add_argument("--bottom_threshold", default="prompt_adaptive_area_loio")
    parser.add_argument("--bottom_negative_source", default="none")
    parser.add_argument("--max_cases", type=int, default=0, help="Only used for bottom_half; 0 renders all selected cases.")
    parser.add_argument("--max_render_dimension", type=int, default=4096)
    parser.add_argument("--class_id", action="append", default=[])
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument(
        "--keep_individual_panels",
        action="store_true",
        help="Keep the eight individual PNG panels. By default only 00_combined_grid.png, legend, and summary are retained.",
    )
    args = parser.parse_args()

    next_root = Path(args.next_root)
    source_root = Path(args.source_root)
    visual_output_root = Path(args.visual_output_root) if args.visual_output_root else next_root
    rows = read_csv(next_root / PRET_DIR / "pret_aaai_next_metrics.csv")
    prompts_path = Path(args.prompts_csv) if args.prompts_csv else source_root / PRET_DIR / "prompts.csv"
    prompts = _dedupe_prompts(read_csv(prompts_path), args.prompt_source)
    if not prompts:
        raise RuntimeError(f"no prompts found for prompt_source={args.prompt_source}: {prompts_path}")
    classes = sorted(args.class_id or sorted({row["class_id"] for row in rows}, key=lambda value: int(value)))
    image_cache: dict[str, dict[str, object]] = {}
    all_cases: list[dict] = []
    if args.selection_mode == "bottom_half":
        selected = _pick_bottom_half_balanced(
            rows,
            classes,
            args.bottom_protocol,
            args.bottom_threshold,
            args.bottom_negative_source,
            args.max_cases,
        )
        for category, metric in selected:
            case_dir = _existing_case_dir(visual_output_root, metric, category, args.render_mode)
            if case_dir is None:
                case_dir = _render_case(
                    source_root,
                    visual_output_root,
                    metric,
                    prompts,
                    prompts,
                    category,
                    args.lambda_neg,
                    args.max_render_dimension,
                    args.keep_individual_panels,
                    args.render_mode,
                    image_cache,
                )
            image_file = "true_prompt_mask.png" if args.render_mode == "gt_prompt" else "00_combined_grid.png"
            all_cases.append({"class_id": metric["class_id"], "category": category, "query_id": metric["query_id"], "dice": metric.get("dice", ""), "path": str(case_dir), "image": str(case_dir / image_file)})
            print(f"visual_summary class={metric['class_id']} category={category} query={metric['query_id']}", flush=True)
    else:
        for class_id in classes:
            selected = []
            selected.extend(_pick_cases(rows, class_id, "1pos", "prompt_adaptive_area_loio", "none", args.limit_per_class))
            selected.extend(_pick_corrections(rows, class_id, args.limit_per_class))
            for category, metric in selected:
                case_dir = _existing_case_dir(visual_output_root, metric, category, args.render_mode)
                if case_dir is None:
                    case_dir = _render_case(
                        source_root,
                        visual_output_root,
                        metric,
                        prompts,
                        prompts,
                        category,
                        args.lambda_neg,
                        args.max_render_dimension,
                        args.keep_individual_panels,
                        args.render_mode,
                        image_cache,
                    )
                image_file = "true_prompt_mask.png" if args.render_mode == "gt_prompt" else "00_combined_grid.png"
                all_cases.append({"class_id": class_id, "category": category, "query_id": metric["query_id"], "dice": metric.get("dice", ""), "path": str(case_dir), "image": str(case_dir / image_file)})
                print(f"visual_summary class={class_id} category={category} query={metric['query_id']}", flush=True)
    if all_cases:
        write_rows = [{key: value for key, value in row.items()} for row in all_cases]
        fieldnames = list(write_rows[0])
        path = visual_output_root / PRET_DIR / "visual_summary" / "visual_summary_index.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(write_rows)
    print(json.dumps({"cases": len(all_cases), "output": str(visual_output_root / PRET_DIR / "visual_summary")}, indent=2))


if __name__ == "__main__":
    main()
