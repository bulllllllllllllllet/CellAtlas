from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_eval_in_context import scores_from_prompt
from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    connected_component_topk_mask,
    mean_std_threshold,
    otsu_score_threshold,
    parse_segment_ids,
    percentile_threshold,
    predict_top_area,
    prompt_relative_threshold,
    read_csv,
    segment_adjacency,
)


PALETTE = np.asarray(
    [
        [240, 0, 0], [245, 150, 0], [255, 255, 255], [110, 40, 120],
        [255, 230, 0], [30, 160, 210], [120, 180, 40], [20, 170, 60],
        [0, 70, 150], [0, 150, 140], [180, 180, 180], [230, 0, 120],
    ],
    dtype=np.uint8,
)
CLASS_NAMES = [
    "tumor_epithelium",
    "tumor_stroma",
    "background",
    "necrosis",
    "normal_gland",
    "normal_stroma",
    "submucosa_serosa",
    "muscle",
    "lymphocyte_aggregate",
    "mucus",
    "fat",
    "blood",
]


def colorize_labels(mask: np.ndarray) -> np.ndarray:
    out = np.full((*mask.shape, 3), 245, dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    out[valid] = PALETTE[mask[valid]]
    return out


def normalize_heat(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.quantile(values, [0.02, 0.98])
    scaled = np.clip((values - low) / max(float(high - low), 1e-8), 0, 1)
    red = (255 * scaled).astype(np.uint8)
    blue = (255 * (1 - scaled)).astype(np.uint8)
    green = (180 * (1 - np.abs(scaled - 0.5) * 2)).astype(np.uint8)
    return np.stack([red, green, blue], axis=1)


def segment_values_to_image(segments: np.ndarray, values: np.ndarray, colors: np.ndarray | None = None) -> np.ndarray:
    if colors is None:
        colors = normalize_heat(values)
    out = np.full((*segments.shape, 3), 245, dtype=np.uint8)
    valid = segments >= 0
    out[valid] = colors[segments[valid]]
    return out


def boundaries_overlay(rgb: np.ndarray, segments: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    boundary = np.zeros(segments.shape, dtype=bool)
    boundary[:, 1:] |= segments[:, 1:] != segments[:, :-1]
    boundary[1:, :] |= segments[1:, :] != segments[:-1, :]
    out[boundary] = [0, 0, 0]
    return out


def mask_from_protocol(
    protocol: str,
    scores: np.ndarray,
    areas: np.ndarray,
    positive_scores: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    if protocol == "top10_area":
        return predict_top_area(scores, areas, 0.10)
    if protocol == "top18_area":
        return predict_top_area(scores, areas, 0.18)
    if protocol == "top20_area":
        return predict_top_area(scores, areas, 0.20)
    if protocol == "otsu":
        return scores >= otsu_score_threshold(scores)
    if protocol == "mean_std_0p5":
        return scores >= mean_std_threshold(scores, 0.5)
    if protocol == "mean_std_1p0":
        return scores >= mean_std_threshold(scores, 1.0)
    if protocol == "percentile_85":
        return scores >= percentile_threshold(scores, 85.0)
    if protocol == "percentile_90":
        return scores >= percentile_threshold(scores, 90.0)
    if protocol == "percentile_95":
        return scores >= percentile_threshold(scores, 95.0)
    if protocol == "prompt_relative_margin_0p05":
        return scores >= prompt_relative_threshold(scores, positive_scores, 0.05)
    if protocol == "prompt_relative_margin_0p10":
        return scores >= prompt_relative_threshold(scores, positive_scores, 0.10)
    if protocol == "cc_top20_keep3":
        return connected_component_topk_mask(scores, areas, segment_adjacency(np.asarray(segments), len(scores)), 0.20, 3)
    raise ValueError(f"unknown mask protocol: {protocol}")


def make_panel(title: str, image: np.ndarray, width: int = 420) -> Image.Image:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    scale = width / pil.width
    pil = pil.resize((width, max(1, int(pil.height * scale))), Image.BILINEAR)
    canvas = Image.new("RGB", (width, pil.height + 28), "white")
    canvas.paste(pil, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(20, 20, 20))
    return canvas


def heat_legend(width: int = 420, height: int = 18) -> Image.Image:
    x = np.linspace(0, 1, width, dtype=np.float32)
    red = (255 * x).astype(np.uint8)
    blue = (255 * (1 - x)).astype(np.uint8)
    green = (180 * (1 - np.abs(x - 0.5) * 2)).astype(np.uint8)
    bar = np.stack([red, green, blue], axis=1)[None, :, :]
    bar = np.repeat(bar, height, axis=0)
    return Image.fromarray(bar, "RGB")


def save_single_panel(
    path: Path,
    title: str,
    image: np.ndarray,
    lines: list[str],
    width: int = 1600,
    add_heat_legend: bool = False,
) -> None:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if pil.width > width:
        scale = width / pil.width
        pil = pil.resize((width, max(1, int(round(pil.height * scale)))), Image.BILINEAR)
    text_height = 34 + 22 * max(1, len(lines))
    legend_height = 34 if add_heat_legend else 0
    canvas = Image.new("RGB", (pil.width, text_height + pil.height + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(10, 10, 10))
    for idx, line in enumerate(lines):
        draw.text((12, 36 + idx * 22), line, fill=(60, 60, 60))
    canvas.paste(pil, (0, text_height))
    if add_heat_legend:
        bar = heat_legend(min(520, pil.width - 24), 18)
        y = text_height + pil.height + 8
        canvas.paste(bar, (12, y))
        draw.text((12, y + 19), "low", fill=(40, 40, 40))
        draw.text((12 + bar.width - 28, y + 19), "high", fill=(40, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def resize_for_render(
    rgb: np.ndarray,
    segments: np.ndarray,
    gt: np.ndarray,
    max_render_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not max_render_dimension or max(segments.shape) <= max_render_dimension:
        return np.asarray(rgb), np.asarray(segments), np.asarray(gt)
    scale = max_render_dimension / max(segments.shape)
    render_size = (
        max(1, int(round(segments.shape[1] * scale))),
        max(1, int(round(segments.shape[0] * scale))),
    )
    render_rgb = np.asarray(Image.fromarray(np.asarray(rgb).astype(np.uint8)).resize(render_size, Image.BILINEAR))
    render_segments = np.asarray(Image.fromarray(np.asarray(segments).astype(np.int32), mode="I").resize(render_size, Image.NEAREST), dtype=np.int32)
    render_gt = np.asarray(Image.fromarray(np.asarray(gt).astype(np.uint8)).resize(render_size, Image.NEAREST))
    return render_rgb, render_segments, render_gt


def visualize_one(
    output_root: Path,
    metric: dict[str, str],
    variant: str,
    negative_strategy: str,
    lambda_neg: float,
    mask_protocol: str,
    output_mode: str,
    max_render_dimension: int,
) -> Path:
    image_id = metric["image_id"]
    query_id = metric["query_id"]
    superpixel_dir = output_root / PRET_DIR / image_id
    rgb = np.load(superpixel_dir / "he_10x_rgb.npy", mmap_mode="r")
    segments = np.load(superpixel_dir / "superpixels.npy", mmap_mode="r")
    gt = np.asarray(np.load(output_root / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r"))
    if gt.shape != segments.shape:
        gt = np.asarray(
            Image.fromarray(gt.astype(np.uint8)).resize(
                (segments.shape[1], segments.shape[0]), Image.NEAREST
            )
        )
    render_rgb, render_segments, render_gt = resize_for_render(
        np.asarray(rgb),
        np.asarray(segments),
        np.asarray(gt),
        max_render_dimension,
    )
    records = read_csv(superpixel_dir / "superpixels.csv")
    prompts = [row for row in read_csv(output_root / PRET_DIR / "prompts.csv") if row["query_id"] == query_id and row["prompt_source"] == metric["prompt_source"] and row["shot"] == metric["shot"]]
    if not prompts:
        raise RuntimeError(f"prompt missing for {query_id}")
    prompt = prompts[0]
    tokens = np.load(superpixel_dir / f"tokens_{variant}.npy", mmap_mode="r")
    positive = parse_segment_ids(prompt["positive_segments"])
    negative = parse_segment_ids(prompt["negative_segments"])
    scores = scores_from_prompt(np.asarray(tokens), positive, negative, negative_strategy, lambda_neg)
    areas = np.asarray([float(row["area_10x_pixels"]) for row in records])
    pred = mask_from_protocol(mask_protocol, scores, areas, scores[np.asarray(positive, dtype=np.int64)], np.asarray(segments))
    segment_labels = np.asarray([int(row["gt_tissue_label"]) for row in records], dtype=np.int64)
    target = segment_labels == int(metric["class_id"])
    pred_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    pred_colors[pred] = [20, 180, 70]
    pred_img = segment_values_to_image(render_segments, np.zeros(len(records)), pred_colors)
    score_img = segment_values_to_image(render_segments, scores)
    prompt_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    prompt_colors[positive] = [0, 80, 255]
    if negative:
        prompt_colors[negative] = [230, 0, 140]
    prompt_img = segment_values_to_image(render_segments, np.zeros(len(records)), prompt_colors)
    error_colors = np.full((len(records), 3), [230, 230, 230], dtype=np.uint8)
    error_colors[pred & target] = [20, 170, 60]
    error_colors[pred & ~target] = [230, 0, 140]
    error_colors[~pred & target] = [90, 90, 90]
    error_img = segment_values_to_image(render_segments, np.zeros(len(records)), error_colors)
    cell_density = np.asarray([float(row["cell_density"]) for row in records], dtype=np.float32)
    cell_img = segment_values_to_image(render_segments, cell_density)
    patch_ids = np.asarray([float(row["patch_index"]) for row in records], dtype=np.float32)
    patch_img = segment_values_to_image(render_segments, patch_ids)
    output_dir = output_root / PRET_DIR / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    ap = float(metric.get("average_precision", 0.0))
    metric_key = {
        "top10_area": "top10_area_dice",
        "top18_area": "top18_area_dice",
        "top20_area": "top20_area_dice",
    }.get(mask_protocol, f"{mask_protocol}_dice")
    dice = float(metric.get(metric_key, metric.get("fixed_threshold_dice", 0.0)))
    class_id = int(metric["class_id"])
    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
    case_name = (
        f"c{class_id:02d}_{query_id}_{metric['prompt_source']}_"
        f"{variant}_{mask_protocol}_ap{ap:.3f}_dice{dice:.3f}"
    )
    if output_mode == "grid":
        panels = [
            make_panel("10x HE + superpixel boundary", boundaries_overlay(render_rgb, render_segments)),
            make_panel("Prompt segments: blue=positive, magenta=negative", prompt_img),
            make_panel("GT tissue mask", colorize_labels(render_gt)),
            make_panel(f"{variant} score heatmap", score_img),
            make_panel(f"Prediction: {mask_protocol}", pred_img),
            make_panel("Error: green TP, magenta FP, gray FN", error_img),
            make_panel("Cell density map", cell_img),
            make_panel("Nearest 1024 patch id map", patch_img),
        ]
        cols = 4
        rows = 2
        width = sum(panel.width for panel in panels[:cols])
        height = sum(max(panel.height for panel in panels[r * cols:(r + 1) * cols]) for r in range(rows)) + 55
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"PRET superpixel visualization | {query_id} | class={class_id} {class_name} | {variant}", fill=(20, 20, 20))
        y = 55
        for r in range(rows):
            x = 0
            row_panels = panels[r * cols:(r + 1) * cols]
            row_height = max(panel.height for panel in row_panels)
            for panel in row_panels:
                canvas.paste(panel, (x, y))
                x += panel.width
            y += row_height
        path = output_dir / f"{case_name}.png"
        canvas.save(path)
        return path

    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    common = [
        f"query={query_id}",
        f"target class={class_id} {class_name} | AP={ap:.3f} | Dice({mask_protocol})={dice:.3f}",
    ]
    save_single_panel(
        case_dir / "01_he_superpixel_boundary.png",
        "01 HE image with superpixel boundaries",
        boundaries_overlay(render_rgb, render_segments),
        common + ["black lines = HE-derived superpixel boundaries"],
    )
    save_single_panel(
        case_dir / "02_prompt_segments.png",
        "02 Prompt superpixels",
        prompt_img,
        common + ["blue = positive prompt segments selected by user box/scribble", "magenta = negative prompt segments if present; gray = not used as prompt"],
    )
    save_single_panel(
        case_dir / "03_gt_tissue_mask.png",
        "03 Ground-truth tissue mask",
        colorize_labels(render_gt),
        common + [f"GDPH palette; target class color is class {class_id} {class_name}", "full class palette is listed in 00_legend.txt"],
    )
    save_single_panel(
        case_dir / "04_similarity_score_heatmap.png",
        "04 Cell/image-token similarity score heatmap",
        score_img,
        common + ["blue = low similarity to prompt prototype; green = middle; red = high similarity", "higher score means more likely retrieved as target-like tissue"],
        add_heat_legend=True,
    )
    save_single_panel(
        case_dir / "05_predicted_mask.png",
        f"05 Predicted target mask: {mask_protocol}",
        pred_img,
        common + ["green = predicted retrieved target superpixels", "gray = not predicted as target"],
    )
    save_single_panel(
        case_dir / "06_error_map.png",
        "06 Error map against GT",
        error_img,
        common + ["green = TP retrieved target; magenta = FP retrieved non-target", "gray = FN missed target; light gray = TN / ignored background"],
    )
    save_single_panel(
        case_dir / "07_cell_density_heatmap.png",
        "07 Cell density heatmap",
        cell_img,
        common + ["blue = low cell density; green = middle; red = high cell density", "used to judge whether cell feature coverage is sparse or dense"],
        add_heat_legend=True,
    )
    save_single_panel(
        case_dir / "08_nearest_patch_id_map.png",
        "08 Nearest 1024-patch id map",
        patch_img,
        common + ["colors encode nearest patch id; abrupt large blocks indicate patch-level assignment influence", "this is diagnostic only, not a tissue class color map"],
        add_heat_legend=True,
    )
    legend_lines = [
        "PRET superpixel visualization legend",
        f"query_id: {query_id}",
        f"image_id: {image_id}",
        f"target_class: {class_id} {class_name}",
        f"variant: {variant}",
        f"mask_protocol: {mask_protocol}",
        f"AP: {ap:.6f}",
        f"Dice: {dice:.6f}",
        "",
        "Eight images:",
        "01 HE with black superpixel boundaries.",
        "02 Prompt segments: blue positive, magenta negative, gray unused.",
        "03 GT tissue mask using GDPH palette.",
        "04 Similarity heatmap: blue low, green middle, red high.",
        "05 Predicted mask: green predicted target, gray not target.",
        "06 Error map: green TP, magenta FP, gray FN, light gray TN/ignored.",
        "07 Cell density: blue low, green middle, red high.",
        "08 Nearest patch id map: diagnostic patch assignment colors.",
        "",
        "GDPH class palette:",
    ]
    legend_lines.extend(
        f"{idx}: {name} rgb={PALETTE[idx].tolist()}" for idx, name in enumerate(CLASS_NAMES)
    )
    (case_dir / "00_legend.txt").write_text("\n".join(legend_lines) + "\n", encoding="utf-8")
    (case_dir / "summary.json").write_text(
        json.dumps(
            {
                "query_id": query_id,
                "image_id": image_id,
                "class_id": class_id,
                "class_name": class_name,
                "variant": variant,
                "mask_protocol": mask_protocol,
                "average_precision": ap,
                "dice": dice,
                "prompt_source": metric["prompt_source"],
                "shot": metric["shot"],
                "positive_segments": len(positive),
                "negative_segments": len(negative),
                "render_shape": list(render_segments.shape),
                "source_shape": list(segments.shape),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PRET superpixel in-context results.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default="image_cell_reg")
    parser.add_argument("--negative_strategy", choices=["negative_median", "negative_bank_max"], default="negative_bank_max")
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--limit_per_class", type=int, default=1)
    parser.add_argument("--max_visualizations", type=int, default=0)
    parser.add_argument("--class_id", action="append", type=int, default=[])
    parser.add_argument("--prompt_source", action="append", default=[])
    parser.add_argument("--scope", default="exclude_prompt_region")
    parser.add_argument("--output_mode", choices=["folder", "grid"], default="folder")
    parser.add_argument(
        "--max_render_dimension",
        type=int,
        default=4096,
        help="Downsample visualization canvas before rendering; metrics still use original tokens. Use 0 to render full size.",
    )
    parser.add_argument(
        "--mask_protocol",
        choices=[
            "top10_area",
            "top18_area",
            "top20_area",
            "otsu",
            "mean_std_0p5",
            "mean_std_1p0",
            "percentile_85",
            "percentile_90",
            "percentile_95",
            "prompt_relative_margin_0p05",
            "prompt_relative_margin_0p10",
            "cc_top20_keep3",
        ],
        default="percentile_90",
    )
    parser.add_argument("--selection", nargs="+", choices=["best", "worst", "median"], default=["best", "worst"])
    args = parser.parse_args()
    output_root = Path(args.output_root)
    rows = [
        row for row in read_csv(output_root / PRET_DIR / "pret_metrics.csv")
        if row["variant"] == args.variant and row["scope"] == args.scope
    ]
    if args.class_id:
        requested_classes = set(args.class_id)
        rows = [row for row in rows if int(row["class_id"]) in requested_classes]
    if args.prompt_source:
        requested_sources = set(args.prompt_source)
        rows = [row for row in rows if row["prompt_source"] in requested_sources]
    selected = []
    for class_id in sorted({int(row["class_id"]) for row in rows}):
        subset = [row for row in rows if int(row["class_id"]) == class_id]
        subset.sort(key=lambda row: float(row["average_precision"]), reverse=True)
        if "best" in args.selection:
            selected.extend(subset[: args.limit_per_class])
        if "median" in args.selection and subset:
            middle = len(subset) // 2
            half = max(0, args.limit_per_class // 2)
            selected.extend(subset[max(0, middle - half): middle + max(1, args.limit_per_class - half)])
        if "worst" in args.selection:
            selected.extend(subset[-args.limit_per_class:])
    if args.max_visualizations > 0:
        selected = selected[: args.max_visualizations]
    paths = [
        str(visualize_one(
            output_root,
            row,
            args.variant,
            args.negative_strategy,
            args.lambda_neg,
            args.mask_protocol,
            args.output_mode,
            args.max_render_dimension,
        ))
        for row in selected
    ]
    print(json.dumps({"visualizations": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
