# 作用：组合 RGB/HSV/chroma 与连通距离约束，可视化脂肪组织漏检和背景误纳入。

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

Image.MAX_IMAGE_PIXELS = None

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
MASK_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/masks")
DEFAULT_VISUAL_ROOT = (
    V3_ROOT
    / "pret_superpixel"
    / "visualizations"
    / "phase_a_combined_fat_background_check"
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


def coverage_rows(candidate: np.ndarray, gt: np.ndarray) -> list[dict[str, object]]:
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


def threshold_rgb(rgb: np.ndarray, white_threshold: int, saturation_threshold: int) -> np.ndarray:
    image = rgb.astype(np.int16, copy=False)
    mean_times3 = image.sum(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    return (mean_times3 < white_threshold * 3) & (chroma > saturation_threshold)


def clean_seed(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[str, int]]:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask & False, {"seed_components": 0, "seed_kept_components": 0, "seed_largest_area": 0}
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels], {
        "seed_components": int(count),
        "seed_kept_components": int(keep.sum()),
        "seed_largest_area": int(sizes[1:].max(initial=0)),
    }


def build_candidates(
    rgb: np.ndarray,
    distance: np.ndarray,
    distance_px: int,
) -> list[tuple[str, str, np.ndarray]]:
    image = rgb.astype(np.int16, copy=False)
    chroma = image.max(axis=2) - image.min(axis=2)
    mean_times3 = image.sum(axis=2)
    hsv = rgb2hsv(rgb.astype(np.float32) / 255.0)
    near = distance <= distance_px

    rgb240 = threshold_rgb(rgb, 240, 0) & near
    rgb242 = threshold_rgb(rgb, 242, 0) & near
    rgb244 = threshold_rgb(rgb, 244, 0) & near
    hsv_loose = (hsv[:, :, 1] >= 0.005) & (hsv[:, :, 2] <= 0.99) & near
    hsv_mid = (hsv[:, :, 1] >= 0.005) & (hsv[:, :, 2] <= 0.97) & near
    chroma2 = (chroma >= 2) & (hsv[:, :, 2] <= 0.99) & near
    chroma5 = (chroma >= 5) & (hsv[:, :, 2] <= 0.99) & near
    stain_any = ((hsv[:, :, 1] >= 0.005) | (chroma >= 2)) & (hsv[:, :, 2] <= 0.99) & near
    pale_fat = (mean_times3 < 248 * 3) & ((hsv[:, :, 1] >= 0.005) | (chroma >= 2)) & near
    rgb_or_stain = rgb242 | stain_any
    rgb_or_pale = rgb242 | pale_fat
    rgb244_or_stain = rgb244 | stain_any

    return [
        ("rgb240_dist", "RGB mean<240 dist", rgb240),
        ("rgb242_dist", "RGB mean<242 dist", rgb242),
        ("rgb244_dist", "RGB mean<244 dist", rgb244),
        ("hsv005_v099_dist", "HSV S>=.005 V<=.99 dist", hsv_loose),
        ("hsv005_v097_dist", "HSV S>=.005 V<=.97 dist", hsv_mid),
        ("chroma2_v099_dist", "chroma>=2 V<=.99 dist", chroma2),
        ("chroma5_v099_dist", "chroma>=5 V<=.99 dist", chroma5),
        ("stain_any_dist", "(HSV S or chroma) dist", stain_any),
        ("pale_fat_dist", "mean<248 and stain dist", pale_fat),
        ("rgb242_or_stain", "RGB242 OR stain", rgb_or_stain),
        ("rgb242_or_pale", "RGB242 OR pale stain", rgb_or_pale),
        ("rgb244_or_stain", "RGB244 OR stain", rgb244_or_stain),
    ]


def process_image(
    image_id: str,
    visual_root: Path,
    analysis_max_dim: int,
    render_max_dim: int,
    distance_px: int,
    seed_min_area: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    image_dir = V3_ROOT / "pret_superpixel" / "multiscale_tokens" / image_id / "medium"
    rgb_full = np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r")
    gt_full = np.load(MASK_ROOT / f"{image_id}_gt_mask.npy", mmap_mode="r")
    analysis_size = target_size(tuple(rgb_full.shape[:2]), analysis_max_dim)
    rgb = resize_rgb(np.asarray(rgb_full), analysis_size)
    gt = resize_mask(np.asarray(gt_full), analysis_size)

    seed_raw = threshold_rgb(rgb, 234, 20)
    seed, seed_meta = clean_seed(seed_raw, seed_min_area)
    distance = ndi.distance_transform_edt(~seed)
    candidates = build_candidates(rgb, distance, distance_px)

    output_dir = visual_root / image_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    panels: list[Image.Image] = []
    render_size = target_size(tuple(rgb.shape[:2]), render_max_dim)
    rgb_small = resize_rgb(rgb, render_size)
    gt_small = resize_mask(gt, render_size)
    for name, title, candidate in candidates:
        class_rows = coverage_rows(candidate, gt)
        background = class_fraction(class_rows, 2)
        fat = class_fraction(class_rows, 10)
        submucosa = class_fraction(class_rows, 6)
        summary = {
            "image_id": image_id,
            "candidate": name,
            "title": title,
            "background_fraction": background,
            "fat_fraction": fat,
            "submucosa_serosa_fraction": submucosa,
            "fat_minus_background": fat - background,
            "distance_px": distance_px,
            **seed_meta,
        }
        rows.append(summary)
        for class_row in class_rows:
            rows.append({**summary, **class_row})
        candidate_small = resize_mask(candidate.astype(np.uint8), render_size) > 0
        subtitle = f"fat={fat:.3f} bg={background:.3f} gap={fat - background:.3f}"
        panels.append(add_header(Image.fromarray(overlay_candidate(rgb_small, candidate_small, gt_small), mode="RGB"), title, subtitle))

    write_csv(
        output_dir / "combined_mask_stats.csv",
        rows,
        [
            "image_id",
            "candidate",
            "title",
            "background_fraction",
            "fat_fraction",
            "submucosa_serosa_fraction",
            "fat_minus_background",
            "distance_px",
            "seed_components",
            "seed_kept_components",
            "seed_largest_area",
            "class_id",
            "class_name",
            "pixels",
            "covered_pixels",
            "covered_fraction",
        ],
    )
    make_grid(panels, columns=4).save(output_dir / "combined_mask_grid.png")
    return [row for row in rows if "class_id" not in row], {
        "image_id": image_id,
        "combined_mask_grid": str(output_dir / "combined_mask_grid.png"),
        "stats_csv": str(output_dir / "combined_mask_stats.csv"),
    }


def summarize(visual_root: Path, all_rows: list[dict[str, object]], index_rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = sorted({str(row["candidate"]) for row in all_rows})
    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows = [row for row in all_rows if row["candidate"] == candidate]
        fat = np.asarray([float(row["fat_fraction"]) for row in rows], dtype=np.float64)
        bg = np.asarray([float(row["background_fraction"]) for row in rows], dtype=np.float64)
        sub = np.asarray([float(row["submucosa_serosa_fraction"]) for row in rows], dtype=np.float64)
        summary_rows.append(
            {
                "candidate": candidate,
                "title": rows[0]["title"],
                "image_count": len(rows),
                "mean_fat_fraction": float(fat.mean()),
                "min_fat_fraction": float(fat.min()),
                "mean_background_fraction": float(bg.mean()),
                "max_background_fraction": float(bg.max()),
                "mean_submucosa_serosa_fraction": float(sub.mean()),
                "mean_fat_minus_background": float((fat - bg).mean()),
                "fat90_image_count": int((fat >= 0.9).sum()),
            }
        )
    summary_rows.sort(key=lambda row: (int(row["fat90_image_count"]), float(row["mean_fat_minus_background"])), reverse=True)
    write_csv(
        visual_root / "combined_summary.csv",
        summary_rows,
        [
            "candidate",
            "title",
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
    write_csv(visual_root / "visual_index.csv", index_rows, ["image_id", "combined_mask_grid", "stats_csv"])
    validation = {
        "passed": len(index_rows) > 0 and all(Path(str(row["combined_mask_grid"])).exists() for row in index_rows),
        "rendered_images": len(index_rows),
        "visual_root": str(visual_root),
        "summary_csv": str(visual_root / "combined_summary.csv"),
        "visual_index_csv": str(visual_root / "visual_index.csv"),
    }
    (visual_root / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize combined RGB/HSV/chroma tissue masks for fat/background separation.")
    parser.add_argument("--visual_root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--analysis_max_dim", type=int, default=5000)
    parser.add_argument("--render_max_dim", type=int, default=1800)
    parser.add_argument("--distance_px", type=int, default=256)
    parser.add_argument("--seed_min_area", type=int, default=50000)
    args = parser.parse_args()

    image_ids = args.image_id or PROBLEM_IMAGE_IDS
    args.visual_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for index, image_id in enumerate(image_ids, start=1):
        rows, index_row = process_image(
            image_id,
            args.visual_root,
            args.analysis_max_dim,
            args.render_max_dim,
            args.distance_px,
            args.seed_min_area,
        )
        all_rows.extend(rows)
        index_rows.append(index_row)
        print(f"rendered {index}/{len(image_ids)} image_id={image_id}", flush=True)
    validation = summarize(args.visual_root, all_rows, index_rows)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise RuntimeError("combined mask visualization validation failed")


if __name__ == "__main__":
    main()
