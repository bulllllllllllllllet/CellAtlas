#!/usr/bin/env python3
"""Render and quantify one muscle WSI before/after centered-context inference."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw

from benchmarks.v4.baseline.tools.render_wsi_oracle_comparison_panels import (
    binary_mask_thumbnail,
    overlay,
    target_mask_thumbnail,
    thumbnail,
    title,
)


METHODS = ("sam", "sam_med2d", "wsi_sam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--wsi-id", required=True)
    parser.add_argument("--new-metadata", type=Path, action="append", required=True)
    parser.add_argument("--old-metadata", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--panel-width", type=int, default=900)
    return parser.parse_args()


def load_metadata(paths: list[Path]) -> dict[str, dict]:
    records = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        method = str(record["method"])
        if method in records:
            raise ValueError(f"duplicate metadata for {method}")
        record["metadata_path"] = str(path)
        records[method] = record
    if set(records) != set(METHODS):
        raise ValueError(f"expected {METHODS}, got {sorted(records)}")
    return records


def prompt_records(frame: pd.DataFrame, canvas_width: int, canvas_height: int) -> list[dict]:
    records = []
    for row in frame.loc[frame["has_target"]].to_dict("records"):
        for column, sign in (("positive_point_10x", "positive"), ("negative_point_10x", "negative")):
            value = row[column]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            x_local, y_local = json.loads(str(value))
            x = float(row["x_10x"]) + float(x_local)
            y = float(row["y_10x"]) + float(y_local)
            if not (0 <= x < canvas_width and 0 <= y < canvas_height):
                raise ValueError(f"{sign} prompt outside canvas: {(x, y)}")
            records.append({"x_10x": x, "y_10x": y, "sign": sign})
    return records


def mark_prompts(image: Image.Image, records: list[dict], canvas_width: int, canvas_height: int) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(2, image.width // 360)
    for record in records:
        x = round(record["x_10x"] * image.width / canvas_width)
        y = round(record["y_10x"] * image.height / canvas_height)
        color = (0, 220, 0) if record["sign"] == "positive" else (240, 0, 0)
        if record["sign"] == "positive":
            draw.line((x - radius, y, x + radius, y), fill=color, width=1)
            draw.line((x, y - radius, x, y + radius), fill=color, width=1)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=1)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=1)


def sheet(panels: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    output = Image.new("RGB", (columns * panels[0].width, rows * panels[0].height), "white")
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % columns) * panel.width, (index // columns) * panel.height))
    return output


def seam_metrics(path: Path, stride: int = 384) -> dict:
    image = pyvips.Image.new_from_file(str(path), page=0, access="sequential")
    if image.bands != 1:
        raise ValueError(f"mask must be single-band: {path}")
    array = np.frombuffer(image.write_to_memory(), np.uint8).reshape(image.height, image.width) > 0
    vertical = np.count_nonzero(array[:, 1:] != array[:, :-1], axis=0)
    horizontal = np.count_nonzero(array[1:, :] != array[:-1, :], axis=1)

    def phases(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray([values[np.arange(len(values)) % stride == phase].mean() for phase in range(stride)])
        top = np.argsort(means)[-5:][::-1]
        return means, top

    x_means, x_top = phases(vertical)
    y_means, y_top = phases(horizontal)
    return {
        "shape": [image.height, image.width],
        "binary_values": sorted(int(v) for v in np.unique(array.astype(np.uint8) * 255)),
        "positive_fraction": float(array.mean()),
        "vertical_top_phases": [
            {"phase": int(i), "mean_transitions": float(x_means[i]), "ratio_to_median": float(x_means[i] / np.median(x_means))}
            for i in x_top
        ],
        "horizontal_top_phases": [
            {"phase": int(i), "mean_transitions": float(y_means[i]), "ratio_to_median": float(y_means[i] / np.median(y_means))}
            for i in y_top
        ],
        "hann_crossover_phase_64_ratio": {
            "vertical": float(x_means[64] / np.median(x_means)),
            "horizontal": float(y_means[64] / np.median(y_means)),
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output_root / f"sam_context_muscle_diagnostic_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tasks = pd.read_parquet(args.task_manifest)
    selected = tasks.loc[tasks["wsi_id"].eq(args.wsi_id)]
    if len(selected) != 1:
        raise ValueError(f"task manifest does not uniquely identify {args.wsi_id}")
    task = selected.iloc[0]
    if int(task.target_class) != 7 or str(task.target_class_name) != "muscle":
        raise ValueError("selected task is not muscle class 7")
    frame = pd.read_parquet(task.patch_prompt_manifest).sort_values("source_index")
    canvas_width = int(frame.iloc[0].output_width_10x)
    canvas_height = int(frame.iloc[0].output_height_10x)
    gt_path = Path(task.gt_path)
    he_path = Path(task.wsi_path)
    gt = pyvips.Image.new_from_file(str(gt_path), access="sequential")
    if (gt.width, gt.height) not in ((canvas_width, canvas_height), (canvas_width - 1, canvas_height - 1)):
        raise ValueError(f"GT/canvas mismatch: GT={(gt.width, gt.height)} canvas={(canvas_width, canvas_height)}")
    prompts = prompt_records(frame, canvas_width, canvas_height)
    if sum(record["sign"] == "positive" for record in prompts) != 413:
        raise ValueError("unexpected positive prompt count")
    if sum(record["sign"] == "negative" for record in prompts) != 369:
        raise ValueError("unexpected negative prompt count")

    new = load_metadata(args.new_metadata)
    old = load_metadata(args.old_metadata)
    for method, metadata in new.items():
        required = {
            "wsi_id": args.wsi_id, "target_class": 7, "target_class_name": "muscle",
            "context_scale": 2, "context_size_10x": 1024,
            "stitched_tile_size_10x": 512, "model_calls": 413,
            "positive_clicks": 413, "negative_clicks": 369, "complete_protocol": True,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"{method} metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}")

    preview_height = max(1, round(args.panel_width * canvas_height / canvas_width))
    he = thumbnail(he_path, args.panel_width, preview_height)
    gt_mask = target_mask_thumbnail(gt_path, (34, 172, 56), args.panel_width, preview_height)
    new_masks = {
        method: binary_mask_thumbnail(Path(metadata["mask_tiff"]), args.panel_width, preview_height)
        for method, metadata in new.items()
    }
    old_masks = {
        method: binary_mask_thumbnail(Path(metadata["mask_tiff"]), args.panel_width, preview_height)
        for method, metadata in old.items()
    }

    overlay_panels = []
    gt_overlay = overlay(he, gt_mask, (34, 172, 56))
    mark_prompts(gt_overlay, prompts, canvas_width, canvas_height)
    overlay_panels.append(title(gt_overlay, "GT muscle | green + positive / red x negative"))
    colors = {"sam": (0, 210, 255), "sam_med2d": (255, 140, 0), "wsi_sam": (220, 60, 220)}
    for method in METHODS:
        panel = overlay(he, new_masks[method], colors[method])
        mark_prompts(panel, prompts, canvas_width, canvas_height)
        overlay_panels.append(title(panel, f"{method} context2 | Dice={float(new[method]['dice']):.4f}"))
    overlay_path = output / f"muscle_context2_overlays_{args.timestamp}.png"
    sheet(overlay_panels, 2).save(overlay_path, optimize=True)

    binary_panels = [title(Image.fromarray(gt_mask.astype(np.uint8) * 255, "L").convert("RGB"), "GT muscle binary")]
    for method in METHODS:
        image = Image.fromarray(new_masks[method].astype(np.uint8) * 255, "L").convert("RGB")
        binary_panels.append(title(image, f"{method} context2 binary | Dice={float(new[method]['dice']):.4f}"))
    binary_path = output / f"muscle_context2_binary_{args.timestamp}.png"
    sheet(binary_panels, 2).save(binary_path, optimize=True)

    before_after_panels = []
    for method in METHODS:
        for label, masks, metadata in (("old tile512", old_masks, old), ("context1024 center512", new_masks, new)):
            panel = overlay(he, masks[method], colors[method])
            mark_prompts(panel, prompts, canvas_width, canvas_height)
            before_after_panels.append(title(panel, f"{method} | {label} | Dice={float(metadata[method]['dice']):.4f}"))
    before_after_path = output / f"muscle_old_vs_context2_{args.timestamp}.png"
    sheet(before_after_panels, 2).save(before_after_path, optimize=True)

    seam = {
        method: {
            "old": seam_metrics(Path(old[method]["mask_tiff"])),
            "context2": seam_metrics(Path(new[method]["mask_tiff"])),
        }
        for method in METHODS
    }
    report = {
        "timestamp": args.timestamp,
        "wsi_id": args.wsi_id,
        "target_class": 7,
        "target_class_name": "muscle",
        "target_rgb": [34, 172, 56],
        "canvas_10x": [canvas_height, canvas_width],
        "prompt_counts": {"positive": 413, "negative": 369},
        "coordinate_conversion": "global_10x = tile_origin_10x + frozen_local_prompt_10x; preview scales both axes independently to the exact canvas extent",
        "visual_review": "pending_user_review",
        "panels": {
            "overlays": str(overlay_path),
            "binary": str(binary_path),
            "old_vs_context2": str(before_after_path),
        },
        "metrics": {
            method: {
                "old_dice": float(old[method]["dice"]),
                "context2_dice": float(new[method]["dice"]),
                "context2_iou": float(new[method]["iou"]),
                "context2_precision": float(new[method]["precision"]),
                "context2_recall": float(new[method]["recall"]),
            }
            for method in METHODS
        },
        "seam_phase_analysis": seam,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "panels": report["panels"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
