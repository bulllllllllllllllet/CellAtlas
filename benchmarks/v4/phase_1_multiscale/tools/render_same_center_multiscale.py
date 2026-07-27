#!/usr/bin/env python3
"""Render model-native same-center fine/middle/coarse HE crops and a WSI thumbnail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--patch-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--thumbnail-max-size", type=int, default=1600)
    return parser.parse_args()


def to_rgb_array(image: pyvips.Image) -> np.ndarray:
    if image.bands < 3:
        raise ValueError("HE image has fewer than three bands")
    image = image[:3].cast("uchar")
    return np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width, image.bands),
    ).copy()


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if len(args.timestamp) != 15 or args.timestamp[8] != "_":
        raise ValueError("--timestamp must use YYYYMMDD_HHMMSS")
    output_dir = args.output_root / f"same_center_multiscale_{args.timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)

    patches = pd.read_parquet(args.patch_index)
    selected = patches.loc[patches["patch_id"] == args.patch_id]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one patch row, found {len(selected)}")
    row = selected.iloc[0]
    slides = pd.read_parquet(args.cohort_manifest)
    selected_slide = slides.loc[slides["wsi_id"] == row["wsi_id"]]
    if len(selected_slide) != 1:
        raise ValueError(f"expected exactly one cohort row, found {len(selected_slide)}")
    slide_row = selected_slide.iloc[0]
    if str(row["wsi_path"]) != str(slide_row["wsi_path"]):
        raise ValueError("patch-index and cohort-manifest WSI paths disagree")

    source = pyvips.Image.new_from_file(str(row["wsi_path"]), access="random")
    expected_dims = (int(slide_row["level0_width"]), int(slide_row["level0_height"]))
    if (source.width, source.height) != expected_dims:
        raise ValueError(f"source dimensions disagree: {(source.width, source.height)} vs {expected_dims}")

    output_size = int(row["width_10x"])
    if output_size != int(row["height_10x"]):
        raise ValueError("model-native output patch must be square")
    fine_x, fine_y = int(row["x_level0"]), int(row["y_level0"])
    fine_w, fine_h = int(row["width_level0"]), int(row["height_level0"])
    fields: list[dict[str, int | str]] = []
    output_dir.mkdir(parents=True)
    colors = {"fine": (220, 20, 60), "middle": (255, 165, 0), "coarse": (30, 144, 255)}
    for name, multiplier in (("fine", 1), ("middle", 2), ("coarse", 4)):
        width, height = fine_w * multiplier, fine_h * multiplier
        x = fine_x - ((multiplier - 1) * fine_w // 2)
        y = fine_y - ((multiplier - 1) * fine_h // 2)
        if x < 0 or y < 0 or x + width > source.width or y + height > source.height:
            raise ValueError(f"{name} field is outside WSI bounds: {(x, y, width, height)}")
        patch = source.crop(x, y, width, height).resize(output_size / width)
        array = to_rgb_array(patch)
        if array.shape != (output_size, output_size, 3):
            raise RuntimeError(f"unexpected {name} output shape: {array.shape}")
        filename = f"{name}_model_input_{output_size}x{output_size}_{args.timestamp}.png"
        save_png(array, output_dir / filename)
        fields.append({
            "scale": name,
            "level0_x": x,
            "level0_y": y,
            "level0_width": width,
            "level0_height": height,
            "output_width": output_size,
            "output_height": output_size,
            "filename": filename,
        })

    thumb_scale = min(args.thumbnail_max_size / source.width, args.thumbnail_max_size / source.height, 1.0)
    thumb_array = to_rgb_array(source.resize(thumb_scale))
    plain_name = f"wsi_thumbnail_{args.timestamp}.png"
    save_png(thumb_array, output_dir / plain_name)
    annotated = Image.fromarray(thumb_array, mode="RGB")
    draw = ImageDraw.Draw(annotated)
    for field in reversed(fields):
        x0 = round(int(field["level0_x"]) * thumb_scale)
        y0 = round(int(field["level0_y"]) * thumb_scale)
        x1 = round((int(field["level0_x"]) + int(field["level0_width"])) * thumb_scale)
        y1 = round((int(field["level0_y"]) + int(field["level0_height"])) * thumb_scale)
        line_width = max(2, round(4 * thumb_scale))
        draw.rectangle((x0, y0, x1, y1), outline=colors[str(field["scale"])], width=line_width)
    annotated_name = f"wsi_thumbnail_with_scale_boxes_{args.timestamp}.png"
    annotated.save(output_dir / annotated_name)

    center = [fine_x + fine_w / 2.0, fine_y + fine_h / 2.0]
    metadata = {
        "timestamp": args.timestamp,
        "source_patch_index": str(args.patch_index),
        "source_cohort_manifest": str(args.cohort_manifest),
        "wsi_id": str(row["wsi_id"]),
        "patch_id": str(row["patch_id"]),
        "split": str(row["split"]),
        "wsi_path": str(row["wsi_path"]),
        "source_level0_dimensions": list(expected_dims),
        "shared_center_level0_xy": center,
        "model_input_contract": "10x/5x/2.5x same-center fields, each resized to 512x512 RGB; training concatenates them to 9x512x512",
        "fields": fields,
        "thumbnail": {"filename": plain_name, "scale_from_level0": thumb_scale, "shape": list(thumb_array.shape)},
        "annotated_thumbnail": annotated_name,
        "box_colors_rgb": {name: list(color) for name, color in colors.items()},
        "programmatic_validation": {
            "source_identity_matches_manifests": True,
            "source_dimensions_match_manifest": True,
            "all_fields_in_bounds": True,
            "all_fields_share_center": all(
                [int(item["level0_x"]) + int(item["level0_width"]) / 2.0,
                 int(item["level0_y"]) + int(item["level0_height"]) / 2.0] == center
                for item in fields
            ),
            "all_patch_outputs_are_model_native_shape": True,
            "human_visual_review": "pending",
        },
    }
    metadata_name = f"metadata_{args.timestamp}.json"
    (output_dir / metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
