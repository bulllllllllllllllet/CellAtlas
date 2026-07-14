# 作用：验证“低倍宽松 tissue mask + 中倍 mask 内 superpixel + 低置信区域高倍复核”的脂肪覆盖效果。

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import sobel

Image.MAX_IMAGE_PIXELS = None

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
MASK_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/masks")
DEFAULT_VISUAL_ROOT = (
    V3_ROOT
    / "pret_superpixel"
    / "visualizations"
    / "phase_a_lowmag_mask_refine_check"
)

PROBLEM_IMAGE_IDS = [
    "445004-23-HE-DX1",
    "446655-13-HE-DX1",
    "1114546-16-HE-DX1",
    "1307108-8-HE-DX1",
    "1333184-7-HE-DX1",
    "1406159-11-HE-DX1",
    "1411869-R1-HE-DX1",
    "1437227-8-HE-DX1",
    "1512244-15-HE-DX1",
    "1723107-14-HE-DX1",
    "1727521-10-HE-DX1",
    "1848629-6-HE-DX1",
    "1850986-8-HE-DX1",
    "1862212-5-HE-DX1",
]

CLASS_NAMES = {
    2: "background",
    6: "submucosa_serosa",
    10: "fat",
}


def target_size(shape: tuple[int, int], max_dim: int) -> tuple[int, int]:
    height, width = shape
    if max_dim <= 0 or max(height, width) <= max_dim:
        return width, height
    scale = max_dim / max(height, width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR), dtype=np.uint8)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST), dtype=np.uint8)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def add_header(panel: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header_h = 58
    out = Image.new("RGB", (panel.width, panel.height + header_h), (245, 245, 245))
    out.paste(panel, (0, header_h))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)
    draw.text((10, 30), subtitle, fill=(70, 70, 70), font=font)
    return out


def make_grid(panels: list[Image.Image], columns: int) -> Image.Image:
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * width, (index // columns) * height))
    return grid


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask & False
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def build_lowmag_mask(
    low_rgb: np.ndarray,
    min_component_area: int,
    close_iterations: int,
    dilate_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = low_rgb.astype(np.int16, copy=False)
    rgb_float = low_rgb.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb_float)
    mean_times3 = image.sum(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    strong_tissue = (mean_times3 < 244 * 3) & ((saturation >= 0.035) | (chroma >= 8))
    pale_tissue = (mean_times3 < 252 * 3) & (value <= 0.998) & ((saturation >= 0.010) | (chroma >= 2))
    edge = np.maximum(sobel(1.0 - value), sobel(-np.log(np.clip(rgb_float, 1 / 255, 1.0)).max(axis=2)))
    edge_binary = edge >= float(np.percentile(edge, 88.0))
    edge_density = ndi.uniform_filter(edge_binary.astype(np.float32), size=7)
    edge_tissue = (edge_density >= 0.020) & (mean_times3 < 253 * 3)

    seed = keep_large_components(strong_tissue | edge_tissue, min_component_area)
    seed = ndi.binary_closing(seed, iterations=close_iterations)
    seed = ndi.binary_fill_holes(seed)
    loose = ndi.binary_dilation(seed, iterations=dilate_iterations) | pale_tissue
    loose = keep_large_components(loose, min_component_area)
    loose = ndi.binary_closing(loose, iterations=max(1, close_iterations // 2))
    loose = ndi.binary_fill_holes(loose)

    high_conf = ndi.binary_dilation(strong_tissue | edge_tissue, iterations=max(1, dilate_iterations // 2)) & loose
    low_conf = loose & ~high_conf
    return loose, high_conf, low_conf


def coverage_row(candidate: np.ndarray, gt: np.ndarray, class_id: int) -> dict[str, object]:
    mask = gt == class_id
    pixels = int(mask.sum())
    covered = int((candidate & mask).sum())
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "pixels": pixels,
        "covered_pixels": covered,
        "covered_fraction": covered / pixels if pixels else 0.0,
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    image = rgb.copy()
    tint = np.zeros_like(image)
    tint[..., 1] = 215
    image[mask] = (0.58 * image[mask] + 0.42 * tint[mask]).astype(np.uint8)
    if gt is not None:
        fat_missed = (gt == 10) & ~mask
        background_hit = (gt == 2) & mask
        image[fat_missed] = [255, 0, 255]
        image[background_hit] = [255, 175, 0]
    return image


def overlay_low_conf(rgb: np.ndarray, low_mask: np.ndarray, low_conf: np.ndarray, gt: np.ndarray) -> np.ndarray:
    image = overlay_mask(rgb, low_mask, gt)
    image[low_conf] = (0.45 * image[low_conf] + 0.55 * np.array([0, 130, 255], dtype=np.float32)).astype(np.uint8)
    return image


def process_image(
    image_id: str,
    visual_root: Path,
    low_max_dim: int,
    analysis_max_dim: int,
    render_max_dim: int,
    min_component_area: int,
    close_iterations: int,
    dilate_iterations: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    image_dir = V3_ROOT / "pret_superpixel" / "multiscale_tokens" / image_id / "medium"
    rgb_full = np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r")
    gt_full = np.load(MASK_ROOT / f"{image_id}_gt_mask.npy", mmap_mode="r")

    low_size = target_size(tuple(rgb_full.shape[:2]), low_max_dim)
    low_rgb = resize_rgb(np.asarray(rgb_full), low_size)
    low_mask, high_conf, low_conf = build_lowmag_mask(low_rgb, min_component_area, close_iterations, dilate_iterations)

    analysis_size = target_size(tuple(rgb_full.shape[:2]), analysis_max_dim)
    rgb = resize_rgb(np.asarray(rgb_full), analysis_size)
    gt = resize_mask(np.asarray(gt_full), analysis_size)
    low_mask_a = resize_mask(low_mask.astype(np.uint8), analysis_size) > 0
    high_conf_a = resize_mask(high_conf.astype(np.uint8), analysis_size) > 0
    low_conf_a = resize_mask(low_conf.astype(np.uint8), analysis_size) > 0

    rows: list[dict[str, object]] = []
    for candidate_name, candidate in [
        ("lowmag_loose_mask", low_mask_a),
        ("lowmag_high_conf", high_conf_a),
        ("highmag_review_low_conf", low_conf_a),
    ]:
        class_rows = [coverage_row(candidate, gt, class_id) for class_id in CLASS_NAMES]
        summary = {
            "image_id": image_id,
            "candidate": candidate_name,
            "low_max_dim": low_max_dim,
            "analysis_max_dim": analysis_max_dim,
            "low_conf_fraction_in_mask": float(low_conf_a.sum() / low_mask_a.sum()) if low_mask_a.any() else 0.0,
        }
        for class_row in class_rows:
            rows.append({**summary, **class_row})

    output_dir = visual_root / image_id
    output_dir.mkdir(parents=True, exist_ok=True)
    render_size = target_size(tuple(rgb.shape[:2]), render_max_dim)
    rgb_small = resize_rgb(rgb, render_size)
    gt_small = resize_mask(gt, render_size)
    low_mask_small = resize_mask(low_mask_a.astype(np.uint8), render_size) > 0
    high_conf_small = resize_mask(high_conf_a.astype(np.uint8), render_size) > 0
    low_conf_small = resize_mask(low_conf_a.astype(np.uint8), render_size) > 0
    gt_focus = np.full((*gt_small.shape, 3), 245, dtype=np.uint8)
    gt_focus[gt_small == 2] = [235, 235, 235]
    gt_focus[gt_small == 10] = [255, 0, 255]
    gt_focus[gt_small == 6] = [0, 180, 80]
    panels = [
        add_header(Image.fromarray(rgb_small, mode="RGB"), "HE"),
        add_header(Image.fromarray(gt_focus, mode="RGB"), "GT focus", "magenta=fat green=submucosa gray=background"),
        add_header(Image.fromarray(overlay_mask(rgb_small, low_mask_small, gt_small), mode="RGB"), "lowmag loose tissue mask", "green=mask magenta=missed fat orange=background hit"),
        add_header(Image.fromarray(overlay_mask(rgb_small, high_conf_small, gt_small), mode="RGB"), "high-confidence tissue", "strong low-mag tissue evidence"),
        add_header(Image.fromarray(overlay_low_conf(rgb_small, low_mask_small, low_conf_small, gt_small), mode="RGB"), "high-mag review candidates", "blue=low-confidence area inside loose mask"),
    ]
    grid = make_grid(panels, columns=3)
    grid_path = output_dir / "lowmag_refine_grid.png"
    grid.save(grid_path)
    write_csv(
        output_dir / "lowmag_refine_stats.csv",
        rows,
        [
            "image_id",
            "candidate",
            "low_max_dim",
            "analysis_max_dim",
            "low_conf_fraction_in_mask",
            "class_id",
            "class_name",
            "pixels",
            "covered_pixels",
            "covered_fraction",
        ],
    )
    return rows, {
        "image_id": image_id,
        "lowmag_refine_grid": str(grid_path),
        "stats_csv": str(output_dir / "lowmag_refine_stats.csv"),
    }


def summarize(visual_root: Path, all_rows: list[dict[str, object]], index_rows: list[dict[str, object]]) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    for candidate in sorted({str(row["candidate"]) for row in all_rows}):
        group = [row for row in all_rows if row["candidate"] == candidate]
        by_class = {int(row["class_id"]): [] for row in group}
        for row in group:
            by_class[int(row["class_id"])].append(float(row["covered_fraction"]))
        fat = np.asarray(by_class.get(10, []), dtype=np.float64)
        bg = np.asarray(by_class.get(2, []), dtype=np.float64)
        sub = np.asarray(by_class.get(6, []), dtype=np.float64)
        summary_rows.append(
            {
                "candidate": candidate,
                "image_count": len(fat),
                "mean_fat_fraction": float(fat.mean()) if len(fat) else 0.0,
                "min_fat_fraction": float(fat.min()) if len(fat) else 0.0,
                "mean_background_fraction": float(bg.mean()) if len(bg) else 0.0,
                "max_background_fraction": float(bg.max()) if len(bg) else 0.0,
                "mean_submucosa_serosa_fraction": float(sub.mean()) if len(sub) else 0.0,
                "fat90_image_count": int((fat >= 0.9).sum()) if len(fat) else 0,
            }
        )
    summary_rows.sort(key=lambda row: (int(row["fat90_image_count"]), float(row["mean_fat_fraction"])), reverse=True)
    write_csv(
        visual_root / "lowmag_refine_summary.csv",
        summary_rows,
        [
            "candidate",
            "image_count",
            "mean_fat_fraction",
            "min_fat_fraction",
            "mean_background_fraction",
            "max_background_fraction",
            "mean_submucosa_serosa_fraction",
            "fat90_image_count",
        ],
    )
    write_csv(visual_root / "visual_index.csv", index_rows, ["image_id", "lowmag_refine_grid", "stats_csv"])
    validation = {
        "passed": len(index_rows) > 0 and all(Path(str(row["lowmag_refine_grid"])).exists() for row in index_rows),
        "rendered_images": len(index_rows),
        "visual_root": str(visual_root),
        "summary_csv": str(visual_root / "lowmag_refine_summary.csv"),
        "visual_index_csv": str(visual_root / "visual_index.csv"),
    }
    (visual_root / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate low-mag loose tissue mask with high-mag review candidates.")
    parser.add_argument("--visual_root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--low_max_dim", type=int, default=1200)
    parser.add_argument("--analysis_max_dim", type=int, default=5000)
    parser.add_argument("--render_max_dim", type=int, default=1800)
    parser.add_argument("--min_component_area", type=int, default=256)
    parser.add_argument("--close_iterations", type=int, default=5)
    parser.add_argument("--dilate_iterations", type=int, default=10)
    args = parser.parse_args()

    args.visual_root.mkdir(parents=True, exist_ok=True)
    image_ids = args.image_id or PROBLEM_IMAGE_IDS
    all_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for index, image_id in enumerate(image_ids, start=1):
        rows, index_row = process_image(
            image_id,
            args.visual_root,
            args.low_max_dim,
            args.analysis_max_dim,
            args.render_max_dim,
            args.min_component_area,
            args.close_iterations,
            args.dilate_iterations,
        )
        all_rows.extend(rows)
        index_rows.append(index_row)
        print(f"rendered {index}/{len(image_ids)} image_id={image_id}", flush=True)
    validation = summarize(args.visual_root, all_rows, index_rows)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise RuntimeError("lowmag mask refine visualization validation failed")


if __name__ == "__main__":
    main()
