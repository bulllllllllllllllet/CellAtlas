# 作用：用 HSV S/V 阈值扫描可视化脂肪组织与背景的 tissue mask 分离效果。

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2hsv

Image.MAX_IMAGE_PIXELS = None

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
MASK_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/masks")
DEFAULT_VISUAL_ROOT = (
    V3_ROOT
    / "pret_superpixel"
    / "visualizations"
    / "phase_a_hsv_fat_background_check"
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


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


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


def resize_bool(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


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


def overlay_candidate(rgb: np.ndarray, candidate: np.ndarray, gt: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    tint = np.zeros_like(image)
    tint[..., 1] = 215
    image[candidate] = (0.58 * image[candidate] + 0.42 * tint[candidate]).astype(np.uint8)
    fat_missed = (gt == 10) & ~candidate
    background_hit = (gt == 2) & candidate
    image[fat_missed] = [255, 0, 255]
    image[background_hit] = [255, 175, 0]
    return image


def gray_panel(values: np.ndarray) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    high = float(np.quantile(values, 0.995))
    low = float(np.quantile(values, 0.005))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray, gray, gray], axis=2), mode="RGB")


def gt_focus_panel(gt: np.ndarray) -> Image.Image:
    image = np.full((*gt.shape, 3), 245, dtype=np.uint8)
    image[gt == 2] = [235, 235, 235]
    image[gt == 10] = [255, 0, 255]
    image[gt == 6] = [0, 180, 80]
    return Image.fromarray(image, mode="RGB")


def coverage_by_class(candidate: np.ndarray, gt: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id, class_name in CLASS_NAMES.items():
        mask = gt == class_id
        pixels = int(mask.sum())
        covered = int((candidate & mask).sum())
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixels": pixels,
                "covered_pixels": covered,
                "covered_fraction": covered / pixels if pixels else 0.0,
            }
        )
    return rows


def class_fraction(rows: list[dict[str, object]], class_id: int) -> float:
    for row in rows:
        if int(row["class_id"]) == class_id:
            return float(row["covered_fraction"])
    return 0.0


def render_channel_check(
    image_id: str,
    rgb: np.ndarray,
    hsv: np.ndarray,
    gt: np.ndarray,
    output_dir: Path,
    render_max_dim: int,
) -> None:
    size = target_size(tuple(rgb.shape[:2]), render_max_dim)
    rgb_small = resize_rgb(rgb, size)
    s_small = np.asarray(Image.fromarray((hsv[:, :, 1] * 255).astype(np.uint8)).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    v_small = np.asarray(Image.fromarray((hsv[:, :, 2] * 255).astype(np.uint8)).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    gt_small = resize_mask(gt, size)
    panels = [
        add_header(Image.fromarray(rgb_small, mode="RGB"), "HE", image_id),
        add_header(gray_panel(s_small), "HSV saturation", "bright = higher S"),
        add_header(gray_panel(v_small), "HSV value", "bright = higher V"),
        add_header(gt_focus_panel(gt_small), "GT focus", "magenta=fat green=submucosa gray=background"),
    ]
    make_grid(panels, columns=2).save(output_dir / "hsv_channels.png")


def render_sv_grid(
    image_id: str,
    rgb: np.ndarray,
    hsv: np.ndarray,
    gt: np.ndarray,
    s_values: list[float],
    v_values: list[float],
    stats: list[dict[str, object]],
    output_dir: Path,
    render_max_dim: int,
) -> None:
    size = target_size(tuple(rgb.shape[:2]), render_max_dim)
    rgb_small = resize_rgb(rgb, size)
    gt_small = resize_mask(gt, size)
    s_small = np.asarray(Image.fromarray((hsv[:, :, 1] * 255).astype(np.uint8)).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    v_small = np.asarray(Image.fromarray((hsv[:, :, 2] * 255).astype(np.uint8)).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    stat_by_key = {
        (float(row["s_min"]), float(row["v_max"])): row
        for row in stats
    }
    panels: list[Image.Image] = []
    for s_min in s_values:
        for v_max in v_values:
            candidate = (s_small >= s_min) & (v_small <= v_max)
            row = stat_by_key[(s_min, v_max)]
            title = f"S>={s_min:g} V<={v_max:g}"
            subtitle = (
                f"fat={float(row['fat_fraction']):.3f} "
                f"bg={float(row['background_fraction']):.3f} "
                f"gap={float(row['fat_minus_background']):.3f}"
            )
            panels.append(add_header(Image.fromarray(overlay_candidate(rgb_small, candidate, gt_small), mode="RGB"), title, subtitle))
    make_grid(panels, columns=len(v_values)).save(output_dir / "hsv_sv_grid.png")


def process_image(
    image_id: str,
    visual_root: Path,
    s_values: list[float],
    v_values: list[float],
    analysis_max_dim: int,
    render_max_dim: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    image_dir = V3_ROOT / "pret_superpixel" / "multiscale_tokens" / image_id / "medium"
    rgb_full = np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r")
    gt_full = np.load(MASK_ROOT / f"{image_id}_gt_mask.npy", mmap_mode="r")
    analysis_size = target_size(tuple(rgb_full.shape[:2]), analysis_max_dim)
    rgb = resize_rgb(np.asarray(rgb_full), analysis_size)
    gt = resize_mask(np.asarray(gt_full), analysis_size)
    hsv = rgb2hsv(rgb.astype(np.float32) / 255.0)

    output_dir = visual_root / image_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict[str, object]] = []
    by_class_rows: list[dict[str, object]] = []
    for s_min in s_values:
        for v_max in v_values:
            candidate = (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] <= v_max)
            class_rows = coverage_by_class(candidate, gt)
            background = class_fraction(class_rows, 2)
            fat = class_fraction(class_rows, 10)
            submucosa = class_fraction(class_rows, 6)
            row = {
                "image_id": image_id,
                "s_min": s_min,
                "v_max": v_max,
                "background_fraction": background,
                "fat_fraction": fat,
                "submucosa_serosa_fraction": submucosa,
                "fat_minus_background": fat - background,
            }
            stats.append(row)
            for class_row in class_rows:
                by_class_rows.append({**row, **class_row})

    write_csv(
        output_dir / "hsv_sv_stats.csv",
        by_class_rows,
        [
            "image_id",
            "s_min",
            "v_max",
            "background_fraction",
            "fat_fraction",
            "submucosa_serosa_fraction",
            "fat_minus_background",
            "class_id",
            "class_name",
            "pixels",
            "covered_pixels",
            "covered_fraction",
        ],
    )
    render_channel_check(image_id, rgb, hsv, gt, output_dir, render_max_dim)
    render_sv_grid(image_id, rgb, hsv, gt, s_values, v_values, stats, output_dir, render_max_dim)
    return stats, {
        "image_id": image_id,
        "hsv_sv_grid": str(output_dir / "hsv_sv_grid.png"),
        "hsv_channels": str(output_dir / "hsv_channels.png"),
        "stats_csv": str(output_dir / "hsv_sv_stats.csv"),
    }


def summarize(
    visual_root: Path,
    all_stats: list[dict[str, object]],
    index_rows: list[dict[str, object]],
    fat_target: float,
) -> dict[str, object]:
    keys = sorted({(float(row["s_min"]), float(row["v_max"])) for row in all_stats})
    summary_rows: list[dict[str, object]] = []
    for s_min, v_max in keys:
        rows = [row for row in all_stats if float(row["s_min"]) == s_min and float(row["v_max"]) == v_max]
        fat = np.asarray([float(row["fat_fraction"]) for row in rows], dtype=np.float64)
        bg = np.asarray([float(row["background_fraction"]) for row in rows], dtype=np.float64)
        sub = np.asarray([float(row["submucosa_serosa_fraction"]) for row in rows], dtype=np.float64)
        gap = fat - bg
        summary_rows.append(
            {
                "s_min": s_min,
                "v_max": v_max,
                "image_count": len(rows),
                "mean_fat_fraction": float(fat.mean()),
                "min_fat_fraction": float(fat.min()),
                "mean_background_fraction": float(bg.mean()),
                "max_background_fraction": float(bg.max()),
                "mean_submucosa_serosa_fraction": float(sub.mean()),
                "mean_fat_minus_background": float(gap.mean()),
                "fat90_image_count": int((fat >= fat_target).sum()),
            }
        )
    summary_rows.sort(key=lambda row: (int(row["fat90_image_count"]), float(row["mean_fat_minus_background"])), reverse=True)
    write_csv(
        visual_root / "summary_sv_candidates.csv",
        summary_rows,
        [
            "s_min",
            "v_max",
            "image_count",
            "mean_fat_fraction",
            "min_fat_fraction",
            "mean_background_fraction",
            "max_background_fraction",
            "mean_submucosa_serosa_fraction",
            "mean_fat_minus_background",
            "fat90_image_count",
        ],
    )
    top_rows = [row for row in summary_rows if float(row["mean_fat_fraction"]) >= fat_target]
    top_rows.sort(key=lambda row: (float(row["mean_background_fraction"]), -float(row["mean_fat_fraction"])))
    write_csv(
        visual_root / "top_candidates.csv",
        top_rows,
        [
            "s_min",
            "v_max",
            "image_count",
            "mean_fat_fraction",
            "min_fat_fraction",
            "mean_background_fraction",
            "max_background_fraction",
            "mean_submucosa_serosa_fraction",
            "mean_fat_minus_background",
            "fat90_image_count",
        ],
    )
    write_csv(
        visual_root / "visual_index.csv",
        index_rows,
        ["image_id", "hsv_sv_grid", "hsv_channels", "stats_csv"],
    )
    validation = {
        "passed": len(index_rows) > 0 and all(Path(str(row["hsv_sv_grid"])).exists() and Path(str(row["hsv_channels"])).exists() for row in index_rows),
        "rendered_images": len(index_rows),
        "visual_root": str(visual_root),
        "summary_csv": str(visual_root / "summary_sv_candidates.csv"),
        "top_candidates_csv": str(visual_root / "top_candidates.csv"),
        "visual_index_csv": str(visual_root / "visual_index.csv"),
    }
    (visual_root / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize HSV S/V tissue mask candidates for fat/background separation.")
    parser.add_argument("--visual_root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--s_values", default="0.005,0.01,0.015,0.02,0.03,0.05")
    parser.add_argument("--v_values", default="0.94,0.95,0.96,0.97,0.98,0.99")
    parser.add_argument("--analysis_max_dim", type=int, default=5000)
    parser.add_argument("--render_max_dim", type=int, default=1800)
    parser.add_argument("--fat_target", type=float, default=0.9)
    args = parser.parse_args()

    image_ids = args.image_id or PROBLEM_IMAGE_IDS
    s_values = parse_floats(args.s_values)
    v_values = parse_floats(args.v_values)
    args.visual_root.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for index, image_id in enumerate(image_ids, start=1):
        stats, index_row = process_image(
            image_id,
            args.visual_root,
            s_values,
            v_values,
            args.analysis_max_dim,
            args.render_max_dim,
        )
        all_stats.extend(stats)
        index_rows.append(index_row)
        print(f"rendered {index}/{len(image_ids)} image_id={image_id}", flush=True)
    validation = summarize(args.visual_root, all_stats, index_rows, args.fat_target)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise RuntimeError("HSV visualization validation failed")


if __name__ == "__main__":
    main()
