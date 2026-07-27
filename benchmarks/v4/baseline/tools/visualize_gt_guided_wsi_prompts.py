#!/usr/bin/env python3
"""Visualize WSI thumbnails, muscle GT, and frozen global/patchwise prompts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pyvips


TARGET_RGB = (34, 172, 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--wsi-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--panel-width", type=int, default=900)
    return parser.parse_args()


def vips_to_pil(image: pyvips.Image) -> Image.Image:
    array = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width, image.bands),
    )
    return Image.fromarray(array[..., :3].copy(), "RGB")


def thumbnail_rgb(path: Path, width: int, height: int) -> Image.Image:
    source = pyvips.Image.new_from_file(str(path), access="sequential").extract_band(0, n=3)
    return vips_to_pil(source.thumbnail_image(width, height=height, size="force", crop="none"))


def gt_thumbnail(path: Path, width: int, height: int) -> tuple[Image.Image, np.ndarray]:
    source = pyvips.Image.new_from_file(str(path), access="sequential").extract_band(0, n=3)
    preview = source.resize(
        width / source.width,
        vscale=height / source.height,
        kernel="nearest",
    )
    if (preview.width, preview.height) != (width, height):
        preview = preview.crop(0, 0, min(width, preview.width), min(height, preview.height))
        preview = preview.embed(0, 0, width, height, extend="copy")
    rgb = np.ndarray(
        buffer=preview.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, 3),
    ).copy()
    mask = np.all(rgb == np.asarray(TARGET_RGB, dtype=np.uint8), axis=2)
    return Image.fromarray(rgb, "RGB"), mask


def gt_overlay(he: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(he, dtype=np.float32).copy()
    color = np.asarray(TARGET_RGB, dtype=np.float32)
    base[mask] = 0.45 * base[mask] + 0.55 * color
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def draw_points(
    image: Image.Image,
    positive: list[tuple[float, float]],
    negative: list[tuple[float, float]],
    canvas_size: tuple[int, int],
    radius: int,
) -> None:
    draw = ImageDraw.Draw(image)
    sx = image.width / canvas_size[0]
    sy = image.height / canvas_size[1]
    for points, fill, outline in (
        (positive, (255, 30, 30), (255, 255, 255)),
        (negative, (30, 100, 255), (255, 255, 255)),
    ):
        for x, y in points:
            px, py = x * sx, y * sy
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=fill,
                outline=outline,
                width=max(1, radius // 2),
            )


def add_label(image: Image.Image, text: str) -> Image.Image:
    bar = 46
    output = Image.new("RGB", (image.width, image.height + bar), "white")
    output.paste(image, (0, bar))
    ImageDraw.Draw(output).text((12, 13), text, fill="black", font=ImageFont.load_default())
    return output


def main() -> None:
    args = parse_args()
    tasks = pd.read_parquet(args.task_manifest).set_index("wsi_id", drop=False)
    output = args.output_root / f"wsi_prompt_visualization_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    audit_rows: list[dict] = []
    overview_rows: list[Image.Image] = []

    for wsi_id in args.wsi_id:
        task = tasks.loc[wsi_id]
        prompts = pd.read_parquet(task["patch_prompt_manifest"])
        gt_source = pyvips.Image.new_from_file(str(task["gt_path"]), access="sequential")
        width = args.panel_width
        height = max(1, round(width * gt_source.height / gt_source.width))
        he = thumbnail_rgb(Path(task["wsi_path"]), width, height)
        _, gt_mask = gt_thumbnail(Path(task["gt_path"]), width, height)
        overlay = gt_overlay(he, gt_mask)
        global_panel = overlay.copy()
        patch_panel = overlay.copy()

        j5 = json.loads(Path(task["j5_prompt_json"]).read_text(encoding="utf-8"))
        downsample = float(prompts.iloc[0]["level0_downsample"])
        global_pos = [(float(p["x"]) / downsample, float(p["y"]) / downsample) for p in j5["positive"]]
        global_neg = [(float(p["x"]) / downsample, float(p["y"]) / downsample) for p in j5["negative"]]
        patch_pos: list[tuple[float, float]] = []
        patch_neg: list[tuple[float, float]] = []
        for row in prompts.loc[prompts["has_target"]].itertuples(index=False):
            pos = json.loads(row.positive_point_10x)
            patch_pos.append((float(row.x_10x) + pos[0], float(row.y_10x) + pos[1]))
            if row.negative_available:
                neg = json.loads(row.negative_point_10x)
                patch_neg.append((float(row.x_10x) + neg[0], float(row.y_10x) + neg[1]))

        canvas = (int(prompts.iloc[0]["output_width_10x"]), int(prompts.iloc[0]["output_height_10x"]))
        draw_points(global_panel, global_pos, global_neg, canvas, radius=8)
        draw_points(patch_panel, patch_pos, patch_neg, canvas, radius=3)
        panels = [
            add_label(he, f"{wsi_id} | HE thumbnail"),
            add_label(global_panel, f"Muscle GT overlay | J5: +{len(global_pos)} / -{len(global_neg)}"),
            add_label(patch_panel, f"Muscle GT overlay | patchwise SAM: +{len(patch_pos)} / -{len(patch_neg)}"),
        ]
        triptych = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
        x = 0
        for panel in panels:
            triptych.paste(panel, (x, 0))
            x += panel.width
        path = output / f"{wsi_id}_he_gt_prompts_{args.timestamp}.png"
        triptych.save(path)
        overview_rows.append(triptych)
        audit_rows.append({
            "wsi_id": wsi_id,
            "canvas_width_10x": canvas[0],
            "canvas_height_10x": canvas[1],
            "j5_positive": len(global_pos),
            "j5_negative": len(global_neg),
            "patchwise_positive": len(patch_pos),
            "patchwise_negative": len(patch_neg),
            "manifest_positive_audit_all": bool(prompts.loc[prompts["has_target"], "positive_audit"].all()),
            "manifest_negative_audit_all": bool(prompts.loc[prompts["has_target"], "negative_audit"].all()),
            "all_points_in_canvas": bool(all(
                0 <= x < canvas[0] and 0 <= y < canvas[1]
                for x, y in global_pos + global_neg + patch_pos + patch_neg
            )),
            "visualization": str(path),
        })

    overview = Image.new(
        "RGB",
        (max(row.width for row in overview_rows), sum(row.height for row in overview_rows)),
        "white",
    )
    y = 0
    for row in overview_rows:
        overview.paste(row, (0, y))
        y += row.height
    overview_path = output / f"three_wsi_he_gt_prompts_overview_{args.timestamp}.png"
    overview.save(overview_path)
    audit_path = output / f"prompt_visualization_audit_{args.timestamp}.json"
    audit_path.write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "overview": str(overview_path), "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
