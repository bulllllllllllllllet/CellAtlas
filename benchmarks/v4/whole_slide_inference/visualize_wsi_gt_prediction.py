#!/usr/bin/env python3
"""Render an exact-coordinate GT-versus-prediction panel for one WSI result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvips
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--he-path", type=Path, required=True)
    parser.add_argument("--class-config", type=Path, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preview-width", type=int, default=1600)
    return parser.parse_args()


def load_target(config_path: Path, target_name: str) -> tuple[int, tuple[int, int, int]]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    matches = [item for item in config["data"]["class_map"] if item["name"] == target_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one class named {target_name!r}, found {len(matches)}")
    item = matches[0]
    rgb = tuple(int(value) for value in item["rgb"])
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"invalid RGB contract for {target_name!r}: {rgb}")
    return int(item["id"]), rgb


def validate_source_and_prompt_contract(
    metadata: dict,
    inference_dir: Path,
    he_path: Path,
    gt_path: Path,
    output_width: int,
    output_height: int,
) -> dict[str, bool | int | str]:
    metadata_he = Path(metadata["wsi_path"]).resolve()
    supplied_he = he_path.resolve()
    if supplied_he != metadata_he:
        raise ValueError(f"HE identity mismatch: metadata={metadata_he}, supplied={supplied_he}")
    wsi_id = str(metadata["wsi_id"])
    if gt_path.stem != wsi_id:
        raise ValueError(f"GT identity mismatch: expected stem {wsi_id!r}, got {gt_path.stem!r}")
    prediction_path = Path(metadata["mask_tiff"]).resolve()
    if prediction_path.parent != inference_dir.resolve():
        raise ValueError(
            f"prediction does not belong to inference directory: {prediction_path.parent} != {inference_dir.resolve()}"
        )
    downsample = float(metadata["level0_downsample"])
    if downsample <= 0:
        raise ValueError(f"invalid level0_downsample: {downsample}")
    for index, record in enumerate(metadata["prompt_records"]):
        x_10x = float(record["x_level0"]) / downsample
        y_10x = float(record["y_level0"]) / downsample
        if not (0 <= x_10x < output_width and 0 <= y_10x < output_height):
            raise ValueError(f"prompt {index} is outside the 10x canvas: {(x_10x, y_10x)}")
    return {
        "he_identity_exact": True,
        "gt_stem_matches_wsi_id": True,
        "prediction_parent_matches_inference_dir": True,
        "prompt_count": len(metadata["prompt_records"]),
        "prompts_within_10x_canvas": True,
        "prompt_conversion": f"10x_xy = level0_xy / {downsample:g}",
    }


def vips_rgb_preview(path: Path, width: int, height: int) -> Image.Image:
    source = pyvips.Image.new_from_file(str(path), access="sequential")
    if source.bands < 3:
        raise ValueError(f"HE image must have at least three bands: {path}")
    preview = source.extract_band(0, n=3).thumbnail_image(
        width,
        height=height,
        size="force",
        crop="none",
        linear=False,
    )
    array = np.ndarray(
        buffer=preview.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, preview.bands),
    )
    return Image.fromarray(array[..., :3].copy(), mode="RGB")


def read_binary_masks(
    gt_path: Path,
    prediction_path: Path,
    expected_width: int,
    expected_height: int,
    target_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | list[int]]]:
    gt = pyvips.Image.new_from_file(str(gt_path), access="sequential")
    prediction = pyvips.Image.new_from_file(str(prediction_path), page=0, access="sequential")
    if (prediction.width, prediction.height) != (expected_width, expected_height):
        raise ValueError(
            f"prediction dimensions {(prediction.width, prediction.height)} do not match "
            f"the 10x inference canvas {(expected_width, expected_height)}"
        )
    width_padding = expected_width - gt.width
    height_padding = expected_height - gt.height
    if width_padding not in (0, 1) or height_padding not in (0, 1):
        raise ValueError(
            f"GT dimensions {(gt.width, gt.height)} are not the inference canvas "
            f"{(expected_width, expected_height)} or its exact floor-rounded extent"
        )
    if gt.bands < 3:
        raise ValueError(f"GT must be RGB/RGBA for exact palette decoding: {gt_path}")
    gt_mask = (
        (gt[0] == target_rgb[0])
        & (gt[1] == target_rgb[1])
        & (gt[2] == target_rgb[2])
    ).cast("uchar")
    if prediction.bands != 1:
        prediction = prediction[0]
    prediction = prediction.crop(0, 0, gt.width, gt.height)
    gt_array = np.frombuffer(gt_mask.write_to_memory(), dtype=np.uint8).reshape(gt.height, gt.width)
    pred_array = np.frombuffer(prediction.cast("uchar").write_to_memory(), dtype=np.uint8).reshape(
        gt.height, gt.width
    )
    extent = {
        "evaluation_shape_10x": [int(gt.height), int(gt.width)],
        "inference_shape_10x": [int(expected_height), int(expected_width)],
        "excluded_right_padding_pixels": int(width_padding),
        "excluded_bottom_padding_pixels": int(height_padding),
        "excluded_padding_pixel_count": int(expected_width * expected_height - gt.width * gt.height),
    }
    return gt_array, pred_array, extent


def exact_binary_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | list[int]]:
    total = gt.size
    gt_count = pred_count = intersection = 0
    gt_values: set[int] = set()
    pred_values: set[int] = set()
    chunk = 20_000_000
    gt_flat, pred_flat = gt.reshape(-1), prediction.reshape(-1)
    for start in range(0, total, chunk):
        gt_part = gt_flat[start : start + chunk]
        pred_part = pred_flat[start : start + chunk]
        gt_values.update(int(value) for value in np.unique(gt_part))
        pred_values.update(int(value) for value in np.unique(pred_part))
        gt_bool = gt_part > 0
        pred_bool = pred_part > 0
        gt_count += int(np.count_nonzero(gt_bool))
        pred_count += int(np.count_nonzero(pred_bool))
        intersection += int(np.count_nonzero(gt_bool & pred_bool))
    if not gt_values <= {0, 255} or not pred_values <= {0, 255}:
        raise ValueError(f"non-binary values: GT={sorted(gt_values)}, prediction={sorted(pred_values)}")
    union = gt_count + pred_count - intersection
    return {
        "gt_values": sorted(gt_values),
        "prediction_values": sorted(pred_values),
        "gt_positive_pixels": gt_count,
        "prediction_positive_pixels": pred_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "gt_positive_fraction": gt_count / total,
        "prediction_positive_fraction": pred_count / total,
        "dice": 2.0 * intersection / (gt_count + pred_count) if gt_count + pred_count else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST)) > 0


def overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    array = np.asarray(base, dtype=np.float32).copy()
    array[mask] = array[mask] * 0.48 + np.asarray(color, dtype=np.float32) * 0.52
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def mark_prompts(
    image: Image.Image,
    prompt_records: list[dict],
    output_width: int,
    output_height: int,
    level0_downsample: float,
) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(7, round(image.width / 180))
    stroke = max(2, round(radius / 3))
    for record in prompt_records:
        x_10x = float(record["x_level0"]) / level0_downsample
        y_10x = float(record["y_level0"]) / level0_downsample
        x = round(x_10x * image.width / output_width)
        y = round(y_10x * image.height / output_height)
        if record["sign"] == "positive":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=stroke)
            draw.line((x - radius, y, x + radius, y), fill=(0, 255, 0), width=stroke)
            draw.line((x, y - radius, x, y + radius), fill=(0, 255, 0), width=stroke)
        elif record["sign"] == "negative":
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), width=stroke)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=(255, 0, 0), width=stroke)
        else:
            raise ValueError(f"unsupported prompt sign: {record['sign']!r}")


def titled(image: Image.Image, title: str) -> Image.Image:
    title_height = 34
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    ImageDraw.Draw(canvas).text((8, 9), title, fill="black", font=ImageFont.load_default())
    return canvas


def main() -> None:
    args = parse_args()
    metadata_path = args.inference_dir / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError(f"inference metadata is not complete: {metadata.get('status')!r}")
    output_height, output_width = (int(value) for value in metadata["output_shape_10x"])
    if args.preview_width <= 0:
        raise ValueError("--preview-width must be positive")
    preview_height = max(1, round(output_height * args.preview_width / output_width))
    preview_size = (args.preview_width, preview_height)
    contract_checks = validate_source_and_prompt_contract(
        metadata,
        args.inference_dir,
        args.he_path,
        args.gt_path,
        output_width,
        output_height,
    )
    target_id, target_rgb = load_target(args.class_config, args.target_class)
    gt, prediction, evaluation_extent = read_binary_masks(
        args.gt_path,
        Path(metadata["mask_tiff"]),
        output_width,
        output_height,
        target_rgb,
    )
    metrics = exact_binary_metrics(gt, prediction)
    he = vips_rgb_preview(args.he_path, *preview_size)
    gt_preview = resize_mask(gt, preview_size)
    pred_preview = resize_mask(prediction, preview_size)

    gt_overlay = overlay(he, gt_preview, target_rgb)
    pred_overlay = overlay(he, pred_preview, (0, 210, 255))
    gt_binary = Image.fromarray(np.where(gt_preview, 255, 0).astype(np.uint8), mode="L").convert("RGB")
    pred_binary = Image.fromarray(np.where(pred_preview, 255, 0).astype(np.uint8), mode="L").convert("RGB")
    for panel in (gt_overlay, pred_overlay, gt_binary, pred_binary):
        mark_prompts(
            panel,
            metadata["prompt_records"],
            output_width,
            output_height,
            float(metadata["level0_downsample"]),
        )

    panels = [
        titled(gt_overlay, f"Ground truth overlay: {args.target_class} (class {target_id}, RGB {target_rgb})"),
        titled(pred_overlay, f"J5 prediction overlay: threshold {metadata['threshold']}"),
        titled(gt_binary, "Ground truth binary mask (white = target)"),
        titled(pred_binary, "J5 predicted binary mask (white = predicted target)"),
    ]
    panel_height = panels[0].height
    sheet = Image.new("RGB", (args.preview_width * 2, panel_height * 2), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * args.preview_width, (index // 2) * panel_height))

    png_path = args.inference_dir / f"wsi_gt_vs_prediction_{args.timestamp}.png"
    report_path = args.inference_dir / f"wsi_gt_vs_prediction_{args.timestamp}.json"
    for path in (png_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    sheet.save(png_path, format="PNG", optimize=True)
    report = {
        "timestamp": args.timestamp,
        "wsi_id": metadata["wsi_id"],
        "target_class": args.target_class,
        "target_class_id": target_id,
        "target_rgb": list(target_rgb),
        "coordinate_space": "10x",
        "canvas_width": output_width,
        "canvas_height": output_height,
        "preview_width": args.preview_width,
        "preview_height": preview_height,
        "panel_layout": [
            "GT overlay | J5 prediction overlay",
            "GT binary mask | J5 predicted binary mask",
        ],
        "prompt_legend": {"positive": "green circle with plus", "negative": "red X"},
        "contract_checks": contract_checks,
        "evaluation_extent": evaluation_extent,
        "inputs": {
            "metadata": str(metadata_path),
            "he": str(args.he_path),
            "gt": str(args.gt_path),
            "prediction": metadata["mask_tiff"],
            "class_config": str(args.class_config),
        },
        "metrics": metrics,
        "artifact": str(png_path),
        "visual_review": "pending_user_review",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"panel": str(png_path), "report": str(report_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
