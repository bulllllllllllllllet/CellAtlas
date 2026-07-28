#!/usr/bin/env python3
"""Render strict six-panel patch comparisons from retained frozen-prompt predictions."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


METHODS = (
    ("Joint model J5", "j5", (0, 200, 255)),
    ("SAM", "sam", (255, 140, 0)),
    ("SAM-Med2D", "sam_med2d", (255, 140, 0)),
    ("WSI-SAM", "wsi_sam", (255, 140, 0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--class-config", type=Path)
    parser.add_argument("--j5-dir", type=Path)
    parser.add_argument("--sam-dir", type=Path)
    parser.add_argument("--sam-med2d-dir", type=Path)
    parser.add_argument("--wsi-sam-dir", type=Path)
    parser.add_argument("--positive-points", type=int, choices=(1, 3, 5))
    parser.add_argument(
        "--compose-grids",
        type=Path,
        nargs="+",
        help="Compose existing 2x3 comparison grids horizontally and exit.",
    )
    parser.add_argument(
        "--grid-labels",
        nargs="+",
        help="Labels for --compose-grids, in the same order.",
    )
    parser.add_argument(
        "--grid-scale",
        type=float,
        default=0.6,
        help="Uniform resize factor for each grid in compose mode (default: 0.6).",
    )
    parser.add_argument(
        "--grid-gap",
        type=int,
        default=24,
        help="Horizontal gap in pixels between resized grids (default: 24).",
    )
    parser.add_argument(
        "--grid-layout",
        choices=("horizontal", "two-row"),
        default="horizontal",
        help="Composition layout: one horizontal row or a 2+1 two-row layout.",
    )
    parser.add_argument(
        "--outer-margin",
        type=int,
        default=24,
        help="White outer margin in pixels in compose mode (default: 24).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def dice(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool) & valid
    truth = np.asarray(truth, dtype=bool) & valid
    denominator = int(prediction.sum()) + int(truth.sum())
    return 1.0 if denominator == 0 else float(2 * (prediction & truth).sum() / denominator)


def mask_canvas(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    array = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    array[np.asarray(mask, dtype=bool)] = color
    return Image.fromarray(array, "RGB")


def mark_prompts(image: Image.Image, positive: np.ndarray, negative: np.ndarray) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(5, image.width // 85)
    width = max(2, image.width // 256)
    for x, y in np.asarray(positive, dtype=np.float32):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255), outline=(0, 190, 0), width=width,
        )
        draw.line((x - radius, y, x + radius, y), fill=(0, 190, 0), width=width)
        draw.line((x, y - radius, x, y + radius), fill=(0, 190, 0), width=width)
    for x, y in np.asarray(negative, dtype=np.float32):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255), outline=(220, 0, 0), width=width,
        )
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=(220, 0, 0), width=width)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=(220, 0, 0), width=width)


def mark_panel_labels(sheet: Image.Image, panel_size: int) -> None:
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(20, panel_size // 18))
    margin = max(10, panel_size // 40)
    for index, label in enumerate(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
        column, row = index % 3, index // 3
        left = column * panel_size + margin
        bottom = (row + 1) * panel_size - margin
        box = draw.textbbox((left, bottom), label, font=font, anchor="ls", stroke_width=1)
        draw.rounded_rectangle(
            (box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2),
            radius=3, fill="white",
        )
        draw.text(
            (left, bottom), label, anchor="ls", fill="black", font=font,
            stroke_width=1, stroke_fill="white",
        )


def prediction_path(directory: Path, occurrence_id: str) -> Path:
    matches = list(directory.glob(f"predictions_*/{occurrence_id}.npz"))
    if len(matches) != 1:
        raise ValueError(f"expected one prediction for {occurrence_id} below {directory}, found {matches}")
    return matches[0]


def compose_grids(args: argparse.Namespace) -> None:
    if not args.compose_grids:
        raise ValueError("--compose-grids requires at least one image")
    if not args.grid_labels or len(args.grid_labels) != len(args.compose_grids):
        raise ValueError("--grid-labels must provide one label per --compose-grids image")
    if not 0.0 < args.grid_scale <= 1.0:
        raise ValueError("--grid-scale must be in (0, 1]")
    if args.grid_gap < 0 or args.outer_margin < 0:
        raise ValueError("--grid-gap and --outer-margin must be non-negative")

    source_grids = []
    for path in args.compose_grids:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_grids.append(Image.open(path).convert("RGB"))
    source_sizes = {image.size for image in source_grids}
    if len(source_sizes) != 1:
        raise ValueError(f"all grids must have the same size, found {sorted(source_sizes)}")

    source_width, source_height = source_grids[0].size
    grid_size = (
        max(1, round(source_width * args.grid_scale)),
        max(1, round(source_height * args.grid_scale)),
    )
    resized = [
        image.resize(grid_size, resample=Image.Resampling.LANCZOS)
        for image in source_grids
    ]
    if args.grid_layout == "horizontal":
        canvas_size = (
            2 * args.outer_margin
            + len(resized) * grid_size[0]
            + (len(resized) - 1) * args.grid_gap,
            2 * args.outer_margin + grid_size[1],
        )
        placements = [
            (args.outer_margin + index * (grid_size[0] + args.grid_gap), args.outer_margin)
            for index in range(len(resized))
        ]
    else:
        if len(resized) != 3:
            raise ValueError("--grid-layout two-row requires exactly three grids")
        canvas_size = (
            2 * args.outer_margin + 2 * grid_size[0] + args.grid_gap,
            2 * args.outer_margin + 2 * grid_size[1] + args.grid_gap,
        )
        placements = [
            (args.outer_margin, args.outer_margin),
            (args.outer_margin + grid_size[0] + args.grid_gap, args.outer_margin),
            (args.outer_margin + (grid_size[0] + args.grid_gap) // 2, args.outer_margin + grid_size[1] + args.grid_gap),
        ]
    canvas = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(24, grid_size[0] // 24))
    label_margin = max(8, grid_size[0] // 96)

    for image, label, (left, top) in zip(resized, args.grid_labels, placements, strict=True):
        canvas.paste(image, (left, top))
        label_xy = (left + label_margin, top + label_margin)
        box = draw.textbbox(label_xy, label, font=font, anchor="la", stroke_width=1)
        padding = max(4, grid_size[0] // 256)
        draw.rounded_rectangle(
            (
                box[0] - padding,
                box[1] - padding,
                box[2] + padding,
                box[3] + padding,
            ),
            radius=padding,
            fill="white",
            outline="black",
            width=max(1, padding // 2),
        )
        draw.text(
            label_xy,
            label,
            anchor="la",
            fill="black",
            font=font,
            stroke_width=1,
            stroke_fill="white",
        )

    output = args.output_root / f"patch_grid_composition_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    image_path = output / f"patch_comparison_{args.grid_layout}_{args.timestamp}.png"
    canvas.save(image_path, optimize=True)
    metadata = {
        "timestamp": args.timestamp,
        "layout": args.grid_layout,
        "source_grids": [str(path) for path in args.compose_grids],
        "source_grid_size": [source_width, source_height],
        "grid_labels": list(args.grid_labels),
        "grid_scale": args.grid_scale,
        "resized_grid_size": list(grid_size),
        "grid_gap": args.grid_gap,
        "outer_margin": args.outer_margin,
        "output_size": list(canvas_size),
        "output": str(image_path),
        "visual_review": "pending_user_review",
    }
    (output / f"metadata_{args.timestamp}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


def validate_render_args(args: argparse.Namespace) -> None:
    required = (
        "manifest",
        "metrics",
        "class_config",
        "j5_dir",
        "sam_dir",
        "sam_med2d_dir",
        "wsi_sam_dir",
        "positive_points",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"render mode requires: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    if args.compose_grids:
        compose_grids(args)
        return
    validate_render_args(args)
    output = args.output_root / f"patch_comparison_visualization_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    manifest = pd.read_parquet(args.manifest).sort_values("occurrence_order").reset_index(drop=True)
    expected = pd.read_parquet(args.metrics)
    config = yaml.safe_load(args.class_config.read_text(encoding="utf-8"))
    class_names = {int(item["id"]): str(item["name"]) for item in config["data"]["class_map"]}
    baseline_dirs = {
        "sam": args.sam_dir,
        "sam_med2d": args.sam_med2d_dir,
        "wsi_sam": args.wsi_sam_dir,
    }
    records: list[dict] = []
    rendered: list[Image.Image] = []

    for row in manifest.itertuples(index=False):
        occurrence_id = str(row.occurrence_id)
        common_occurrence_id = occurrence_id.removesuffix(f"_p{args.positive_points}n1")
        episode_index = int(row.episode_index)
        occurrence_order = int(row.occurrence_order)
        arrays = {
            key: np.load(prediction_path(directory, occurrence_id))
            for key, directory in baseline_dirs.items()
        }
        j5_path = args.j5_dir / f"prediction_arrays_{occurrence_order:06d}.npz"
        if not j5_path.is_file():
            raise FileNotFoundError(j5_path)
        arrays["j5"] = np.load(j5_path)

        reference = arrays["sam"]
        image = np.asarray(reference["image"], dtype=np.uint8)
        truth = np.asarray(reference["target_mask"], dtype=bool)
        valid = np.asarray(reference["valid_mask"], dtype=bool)
        positive = np.asarray(reference["positive_points"], dtype=np.float32)
        negative = np.asarray(reference["negative_points"], dtype=np.float32)
        if image.shape != (512, 512, 3) or truth.shape != (512, 512) or valid.shape != truth.shape:
            raise ValueError(f"unexpected retained array shape for {occurrence_id}")
        if len(positive) != args.positive_points or len(negative) != 1:
            raise ValueError(
                f"{occurrence_id} is not an exact {args.positive_points}+1 occurrence"
            )
        for key, data in arrays.items():
            if not np.array_equal(np.asarray(data["target_mask"], dtype=bool), truth):
                raise ValueError(f"target mask mismatch: {occurrence_id}, {key}")
            if not np.array_equal(np.asarray(data["positive_points"], dtype=np.float32), positive):
                raise ValueError(f"positive prompt mismatch: {occurrence_id}, {key}")
            if not np.array_equal(np.asarray(data["negative_points"], dtype=np.float32), negative):
                raise ValueError(f"negative prompt mismatch: {occurrence_id}, {key}")
            prediction = np.asarray(data["binary_mask"])
            if prediction.shape != truth.shape or prediction.dtype != np.bool_:
                raise ValueError(f"prediction must be bool 512x512: {occurrence_id}, {key}")

        panels = [Image.fromarray(image, "RGB"), mask_canvas(truth, (235, 0, 25))]
        method_metrics: dict[str, float] = {}
        for method, key, color in METHODS:
            prediction = np.asarray(arrays[key]["binary_mask"], dtype=bool)
            method_metrics[method] = dice(prediction, truth, valid)
            panels.append(mask_canvas(prediction, color))
        for panel in panels:
            mark_prompts(panel, positive, negative)

        sheet = Image.new("RGB", (512 * 3, 512 * 2), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, ((index % 3) * 512, (index // 3) * 512))
        mark_panel_labels(sheet, 512)
        class_id = int(row.target_class)
        path = output / (
            f"class_{class_id:02d}_{class_names[class_id]}__{str(row.patch_id).replace('/', '_')}.png"
        )
        sheet.save(path, optimize=True)
        rendered.append(sheet)

        expected_rows = expected[
            (expected["common_occurrence_id"].astype(str) == common_occurrence_id)
            & (expected["positive_points"] == args.positive_points)
        ]
        expected_by_method = expected_rows.set_index("method")["episode_dice"].to_dict()
        if set(expected_by_method) != {item[0] for item in METHODS}:
            raise ValueError(f"missing expected metrics for {occurrence_id}: {sorted(expected_by_method)}")
        deltas = {
            method: abs(value - float(expected_by_method[method]))
            for method, value in method_metrics.items()
        }
        if max(deltas.values()) > 1e-6:
            raise ValueError(f"replayed Dice mismatch for {occurrence_id}: {deltas}")
        records.append({
            "occurrence_id": occurrence_id,
            "common_occurrence_id": common_occurrence_id,
            "episode_index": episode_index,
            "occurrence_order": occurrence_order,
            "patch_id": str(row.patch_id),
            "wsi_id": str(row.wsi_id),
            "target_class": class_id,
            "target_class_name": class_names[class_id],
            "positive_points": positive.tolist(),
            "negative_points": negative.tolist(),
            "dice": method_metrics,
            "expected_dice_delta": deltas,
            "panel": str(path),
            "visual_review": "pending_user_review",
        })

    contact_sheet = Image.new("RGB", (1536, 1024 * len(rendered)), "white")
    for index, image in enumerate(rendered):
        contact_sheet.paste(image, (0, index * 1024))
    contact_path = output / f"patch_comparison_contact_sheet_{args.timestamp}.png"
    contact_sheet.save(contact_path, optimize=True)
    report = {
        "timestamp": args.timestamp,
        "prompt_budget": f"{args.positive_points}+1",
        "panel_layout": [
            "(a) H&E", "(b) target-class GT", "(c) FewClick",
            "(d) SAM", "(e) SAM-Med2D", "(f) WSI-SAM",
        ],
        "prompt_legend": "positive=green plus; negative=red X; identical on all panels",
        "manifest": str(args.manifest),
        "metrics": str(args.metrics),
        "contact_sheet": str(contact_path),
        "cases": records,
    }
    (output / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "cases": len(records), "contact_sheet": str(contact_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
