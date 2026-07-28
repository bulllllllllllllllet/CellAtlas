#!/usr/bin/env python3
"""Render GT-only WSI comparison panels for every tissue class.

Each class selects the WSI where J5 oracle-5 has the largest Dice advantage
over the strongest available SAM variant.  Each panel then displays all three
SAM variants beside J5. The result is explicitly an oracle upper-bound
visualization, not a single-click comparison.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import yaml
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--class-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--target-class", action="append", default=[], help="render only named class(es)")
    parser.add_argument("--case", action="append", default=[], metavar="WSI_ID:CLASS_NAME",
                        help="render an explicitly selected WSI/class case")
    return parser.parse_args()


def pil_from_vips(image: pyvips.Image) -> Image.Image:
    bands = min(image.bands, 3)
    image = image.extract_band(0, n=bands)
    array = np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8, shape=(image.height, image.width, bands))
    if bands == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array[..., :3].copy(), "RGB")


def thumbnail(path: Path, width: int, height: int, nearest: bool = False) -> Image.Image:
    image = pyvips.Image.new_from_file(str(path), page=0, access="sequential")
    if nearest:
        scale = width / image.width
        image = image.resize(scale, vscale=height / image.height, kernel="nearest")
        if image.width != width or image.height != height:
            image = image.crop(0, 0, min(image.width, width), min(image.height, height)).embed(0, 0, width, height)
    else:
        image = image.extract_band(0, n=min(image.bands, 3)).thumbnail_image(
            width, height=height, size="force", crop="none", linear=False
        )
    return pil_from_vips(image)


def target_mask_thumbnail(gt_path: Path, rgb: tuple[int, int, int], width: int, height: int) -> np.ndarray:
    gt = pyvips.Image.new_from_file(str(gt_path), access="sequential")
    if gt.bands < 3:
        raise ValueError(f"GT must have RGB palette channels: {gt_path}")
    mask = ((gt[0] == rgb[0]) & (gt[1] == rgb[1]) & (gt[2] == rgb[2])).cast("uchar")
    preview = mask.resize(width / mask.width, vscale=height / mask.height, kernel="nearest")
    if preview.width != width or preview.height != height:
        preview = preview.crop(0, 0, min(preview.width, width), min(preview.height, height)).embed(0, 0, width, height)
    return np.ndarray(buffer=preview.write_to_memory(), dtype=np.uint8, shape=(height, width)) > 0


def binary_mask_thumbnail(path: Path, width: int, height: int) -> np.ndarray:
    image = pyvips.Image.new_from_file(str(path), page=0, access="sequential")
    if image.bands != 1:
        image = image[0]
    preview = image.resize(width / image.width, vscale=height / image.height, kernel="nearest")
    if preview.width != width or preview.height != height:
        preview = preview.crop(0, 0, min(preview.width, width), min(preview.height, height)).embed(0, 0, width, height)
    values = np.unique(np.frombuffer(preview.write_to_memory(), dtype=np.uint8))
    if not set(values.tolist()) <= {0, 255}:
        raise ValueError(f"prediction is not binary: {path}; values={values.tolist()}")
    return np.ndarray(buffer=preview.write_to_memory(), dtype=np.uint8, shape=(height, width)) > 0


def overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    array = np.asarray(base, dtype=np.float32).copy()
    array[mask] = 0.48 * array[mask] + 0.52 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def mask_canvas(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """Render a binary mask without using the H&E image as a background."""
    array = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    array[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def mark_panel_labels(sheet: Image.Image, panel_width: int, panel_height: int) -> None:
    """Add row-major panel markers (a)--(f) at each panel's lower-left corner."""
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(24, panel_width // 28))
    margin = max(12, panel_width // 50)
    labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for index, label in enumerate(labels):
        column, row = index % 3, index // 3
        left = column * panel_width + margin
        bottom = (row + 1) * panel_height - margin
        box = draw.textbbox((left, bottom), label, font=font, anchor="ls", stroke_width=1)
        draw.rounded_rectangle(
            (box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), radius=3, fill="white"
        )
        draw.text((left, bottom), label, anchor="ls", fill="black", font=font, stroke_width=1, stroke_fill="white")


def mark_j5_prompts(image: Image.Image, metadata: dict, canvas_width: int, canvas_height: int) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(5, image.width // 180)
    downsample = float(metadata["level0_downsample"])
    for record in metadata["prompt_records"]:
        x = float(record["x_level0"]) / downsample
        y = float(record["y_level0"]) / downsample
        if not (0 <= x < canvas_width and 0 <= y < canvas_height):
            raise ValueError(f"prompt outside J5 canvas: {(x, y)}")
        x = round(x * image.width / canvas_width)
        y = round(y * image.height / canvas_height)
        color = (0, 210, 0) if record["sign"] == "positive" else (230, 0, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        if record["sign"] == "positive":
            draw.line((x - radius, y, x + radius, y), fill=color, width=2)
            draw.line((x, y - radius, x, y + radius), fill=color, width=2)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)


def class_map(path: Path) -> dict[str, tuple[int, tuple[int, int, int]]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(item["name"]): (int(item["id"]), tuple(int(v) for v in item["rgb"])) for item in config["data"]["class_map"]}


def main() -> None:
    args = parse_args()
    if args.panel_width <= 0:
        raise ValueError("panel width must be positive")
    output = args.output_root / f"wsi_oracle_class_panels_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Whole-slide TIFFs are large. Keep libvips from retaining previous cases
    # while rendering the 12-class panel set in one process.
    pyvips.cache_set_max(0)
    pyvips.cache_set_max_mem(0)
    pyvips.cache_set_max_files(0)
    classes = class_map(args.class_config)
    metrics = pd.read_parquet(args.summary)
    required = {"method", "wsi_id", "target_class_name", "dice", "mask_tiff", "source_metadata", "candidate_index"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"summary lacks required columns: {sorted(missing)}")
    rows: list[dict] = []
    explicit_cases: dict[str, str] = {}
    for value in args.case:
        if ":" not in value:
            raise ValueError(f"--case must be WSI_ID:CLASS_NAME, got {value!r}")
        wsi_id, target_name = value.split(":", 1)
        if target_name in explicit_cases:
            raise ValueError(f"duplicate explicit class selection: {target_name}")
        explicit_cases[target_name] = wsi_id
    if explicit_cases:
        unknown = sorted(set(explicit_cases) - set(classes))
        if unknown:
            raise ValueError(f"unknown explicit class(es): {unknown}")
        metrics = metrics.loc[metrics["target_class_name"].isin(explicit_cases)]
    if args.target_class:
        unknown = sorted(set(args.target_class) - set(classes))
        if unknown:
            raise ValueError(f"unknown target class(es): {unknown}")
        metrics = metrics.loc[metrics["target_class_name"].isin(args.target_class)]
    for target_name, group in metrics.groupby("target_class_name", sort=True):
        if target_name not in classes:
            raise ValueError(f"class absent from config: {target_name}")
        pivot = group.pivot(index="wsi_id", columns="method", values="dice")
        if "j5_oracle5" not in pivot:
            raise ValueError(f"J5 is missing for class {target_name}")
        baselines = [name for name in ("sam", "sam_med2d", "wsi_sam") if name in pivot]
        pivot["best_baseline"] = pivot[baselines].max(axis=1)
        pivot["gain"] = pivot["j5_oracle5"] - pivot["best_baseline"]
        wsi_id = explicit_cases.get(target_name, str(pivot["gain"].idxmax()))
        if wsi_id not in pivot.index:
            raise ValueError(f"requested WSI {wsi_id!r} is unavailable for class {target_name!r}")
        selected = pivot.loc[wsi_id]
        best_method = str(selected[baselines].idxmax())
        j5 = group.loc[(group.wsi_id == wsi_id) & (group.method == "j5_oracle5")].iloc[0]
        sam_rows = {
            method: group.loc[(group.wsi_id == wsi_id) & (group.method == method)].iloc[0]
            for method in baselines
        }
        has_candidate_manifest = isinstance(j5.prompt_json, str) and j5.prompt_json
        candidate_dir = (
            Path(j5.source_metadata).parent / f"candidate_{int(j5.candidate_index):02d}"
            if has_candidate_manifest else Path(j5.source_metadata).parent
        )
        metadata_path = candidate_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("wsi_id") != wsi_id or metadata.get("status") != "complete":
            raise ValueError(f"invalid J5 metadata for {target_name}: {metadata_path}")
        prediction_path = Path(metadata["mask_tiff"])
        if has_candidate_manifest:
            manifest_paths = list(Path(j5.prompt_json).parent.glob("candidate_manifest_*.parquet"))
            if len(manifest_paths) != 1:
                raise ValueError(f"expected one candidate manifest beside {j5.prompt_json}, found {manifest_paths}")
            candidate_row = pd.read_parquet(manifest_paths[0]).loc[
                lambda frame: (frame["wsi_id"] == wsi_id) & (frame["candidate_index"] == int(j5.candidate_index))
            ]
            if len(candidate_row) != 1:
                raise ValueError(f"candidate manifest does not uniquely identify {wsi_id} candidate {j5.candidate_index}")
            gt_path = Path(candidate_row.iloc[0].gt_path)
            he_path = Path(candidate_row.iloc[0].wsi_path)
        else:
            source_report = json.loads(Path(j5.source_metadata).read_text(encoding="utf-8"))
            gt_path = Path(source_report["inputs"]["gt"])
            he_path = Path(source_report["inputs"]["wsi"])
        target_id, rgb = classes[target_name]
        canvas_height, canvas_width = (int(v) for v in metadata["output_shape_10x"])
        gt = pyvips.Image.new_from_file(str(gt_path), access="sequential")
        if gt_path.stem != wsi_id or gt.width not in (canvas_width, canvas_width - 1) or gt.height not in (canvas_height, canvas_height - 1):
            raise ValueError(f"GT/canvas mismatch for {wsi_id}: GT={(gt.width, gt.height)}, canvas={(canvas_width, canvas_height)}")
        preview_height = max(1, round(args.panel_width * gt.height / gt.width))
        he_panel = thumbnail(he_path, args.panel_width, preview_height)
        gt_mask = target_mask_thumbnail(gt_path, rgb, args.panel_width, preview_height)
        j5_mask = binary_mask_thumbnail(prediction_path, args.panel_width, preview_height)
        gt_panel = mask_canvas(gt_mask, rgb)
        j5_panel = mask_canvas(j5_mask, (0, 210, 255))
        mark_j5_prompts(j5_panel, metadata, canvas_width, canvas_height)
        sam_panels = {
            method: mask_canvas(
                binary_mask_thumbnail(Path(row.mask_tiff), args.panel_width, preview_height),
                (255, 140, 0),
            )
            for method, row in sam_rows.items()
        }
        panels = [
            he_panel,
            gt_panel,
            j5_panel,
            sam_panels["sam"],
            sam_panels["sam_med2d"],
            sam_panels["wsi_sam"],
        ]
        sheet = Image.new("RGB", (args.panel_width * 3, panels[0].height * 2), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, ((index % 3) * args.panel_width, (index // 3) * panels[0].height))
        mark_panel_labels(sheet, args.panel_width, panels[0].height)
        panel_path = output / f"class_{target_id:02d}_{target_name}__{wsi_id}.png"
        sheet.save(panel_path, optimize=True)
        rows.append({
            "target_class": target_id, "target_class_name": target_name, "wsi_id": wsi_id,
            "j5_oracle5_dice": float(j5.dice), "best_baseline_method": best_method,
            "best_baseline_dice": float(selected["best_baseline"]), "dice_gain": float(j5.dice - selected["best_baseline"]),
            "sam_dice": float(sam_rows["sam"].dice),
            "sam_med2d_dice": float(sam_rows["sam_med2d"].dice),
            "wsi_sam_dice": float(sam_rows["wsi_sam"].dice),
            "j5_candidate_index": int(j5.candidate_index), "gt_path": str(gt_path),
            "he_path": str(he_path),
            "j5_mask": str(prediction_path),
            "sam_mask": str(sam_rows["sam"].mask_tiff),
            "sam_med2d_mask": str(sam_rows["sam_med2d"].mask_tiff),
            "wsi_sam_mask": str(sam_rows["wsi_sam"].mask_tiff), "panel": str(panel_path),
        })
        del gt, he_panel, gt_mask, j5_mask, gt_panel, j5_panel, sam_panels, panels, sheet
        gc.collect()
    selection = pd.DataFrame(rows).sort_values("target_class")
    selection.to_parquet(output / "selected_cases.parquet", index=False)
    report = {
        "timestamp": args.timestamp,
        "selection_rule": "per class: largest J5 oracle-5 Dice gain over the strongest external baseline",
        "comparison_scope": "six panels with lower-left row-major markers (a)--(f): H&E, target-class GT, J5 best-of-five GT-selected candidate, SAM, SAM-Med2D, and WSI-SAM GT-guided patchwise masks",
        "prompt_legend": "J5 positive=green plus; J5 negative=red X",
        "visual_review": "pending_user_review",
        "selected_cases": rows,
    }
    (output / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
