# 作用：扫描 HE tissue mask 的白度/饱和度阈值与距离连通约束，定位浅色组织覆盖和背景误纳入的折中点。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_a.visual_check_multiscale import load_gt_mask, load_palette, read_csv, rgb_to_class_mask


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
DATA_MANIFEST = V3_ROOT / "pret_superpixel" / "data_manifest_v3.csv"
PALETTE_JSON = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/gdph_tissue_palette.json")
DEFAULT_VISUAL_ROOT = (
    V3_ROOT
    / "pret_superpixel"
    / "visualizations"
    / "phase_a_multiscale_check_threshold234_inprogress"
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_class_names(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item["name"] for item in payload["classes"]}


def tissue_mask(rgb: np.ndarray, white_threshold: int, saturation_threshold: int) -> np.ndarray:
    image = rgb.astype(np.int16, copy=False)
    mean_times3 = image.sum(axis=2)
    saturation = image.max(axis=2) - image.min(axis=2)
    return (mean_times3 < white_threshold * 3) & (saturation > saturation_threshold)


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[str, int]]:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask & False, {"component_count": 0, "kept_component_count": 0, "largest_component_area": 0}
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    cleaned = keep[labels]
    return cleaned, {
        "component_count": int(count),
        "kept_component_count": int(keep.sum()),
        "largest_component_area": int(sizes[1:].max(initial=0)),
    }


def coverage_rows(
    mask: np.ndarray,
    gt: np.ndarray,
    class_names: dict[int, str],
    mask_name: str,
    params: dict[str, object],
    totals: np.ndarray,
) -> list[dict[str, object]]:
    covered = np.bincount(gt[mask].ravel(), minlength=len(totals))
    rows: list[dict[str, object]] = []
    for class_id in np.flatnonzero(totals):
        pixels = int(totals[class_id])
        covered_pixels = int(covered[class_id])
        rows.append(
            {
                "mask_name": mask_name,
                "white_threshold": params.get("white_threshold", ""),
                "saturation_threshold": params.get("saturation_threshold", ""),
                "distance_px": params.get("distance_px", ""),
                "seed_white_threshold": params.get("seed_white_threshold", ""),
                "seed_saturation_threshold": params.get("seed_saturation_threshold", ""),
                "seed_min_area": params.get("seed_min_area", ""),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "pixels": pixels,
                "covered_pixels": covered_pixels,
                "covered_fraction": covered_pixels / pixels,
            }
        )
    return rows


def class_fraction(rows: list[dict[str, object]], class_name: str) -> float:
    for row in rows:
        if row["class_name"] == class_name:
            return float(row["covered_fraction"])
    return 0.0


def resize_bool(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR), dtype=np.uint8)


def target_size(shape: tuple[int, int], max_dim: int) -> tuple[int, int]:
    height, width = shape
    if max_dim <= 0 or max(height, width) <= max_dim:
        return width, height
    scale = max_dim / max(height, width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def make_panel(rgb_small: np.ndarray, mask_small: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = rgb_small.copy()
    overlay = np.zeros_like(image)
    overlay[..., 1] = 210
    image[mask_small] = (0.58 * image[mask_small] + 0.42 * overlay[mask_small]).astype(np.uint8)
    panel = Image.fromarray(image, mode="RGB")
    header_h = 58
    out = Image.new("RGB", (panel.width, panel.height + header_h), (245, 245, 245))
    out.paste(panel, (0, header_h))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)
    draw.text((10, 30), subtitle, fill=(70, 70, 70), font=font)
    return out


def write_visual(
    path: Path,
    rgb: np.ndarray,
    masks: list[tuple[str, str, np.ndarray]],
    max_dim: int,
) -> None:
    scale = min(1.0, max_dim / max(rgb.shape[:2]))
    size = (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale))))
    rgb_small = resize_rgb(rgb, size)
    panels = [make_panel(rgb_small, resize_bool(mask, size), title, subtitle) for title, subtitle, mask in masks]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", default="445004-23-HE-DX1")
    parser.add_argument("--visual_root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--white_thresholds", default="240,242,244,246,248,250,252")
    parser.add_argument("--saturation_thresholds", default="20,15,10,5,0")
    parser.add_argument("--distance_px", default="0,32,64,96,128,192,256,384,512")
    parser.add_argument("--seed_white_threshold", type=int, default=234)
    parser.add_argument("--seed_saturation_threshold", type=int, default=20)
    parser.add_argument("--seed_min_area", type=int, default=50000)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_dim", type=int, default=1800)
    parser.add_argument(
        "--analysis_max_dim",
        type=int,
        default=0,
        help="Resize HE/GT to this max dimension before scanning. Use 0 for full resolution.",
    )
    args = parser.parse_args()

    rows = read_csv(DATA_MANIFEST)
    row_by_id = {row["image_id"]: row for row in rows}
    if args.image_id not in row_by_id:
        raise SystemExit(f"image_id not found in manifest: {args.image_id}")
    row = row_by_id[args.image_id]

    image_dir = V3_ROOT / "pret_superpixel" / "multiscale_tokens" / args.image_id / "medium"
    rgb_full = np.load(image_dir / "he_10x_rgb.npy", mmap_mode="r")
    analysis_size = target_size(tuple(rgb_full.shape[:2]), args.analysis_max_dim)
    if analysis_size == (rgb_full.shape[1], rgb_full.shape[0]):
        rgb = np.asarray(rgb_full)
    else:
        rgb = np.asarray(Image.fromarray(np.asarray(rgb_full)).resize(analysis_size, Image.BILINEAR), dtype=np.uint8)
    palette = load_palette(PALETTE_JSON)
    gt = load_gt_mask(row, (rgb.shape[1], rgb.shape[0]), V3_ROOT)
    if gt.ndim == 3:
        gt = rgb_to_class_mask(gt.astype(np.uint8), palette)
    gt = np.asarray(gt, dtype=np.int32)

    class_names = load_class_names(PALETTE_JSON)
    totals = np.bincount(gt.ravel(), minlength=256)
    output_dir = args.visual_root / args.image_id / "connectivity_tissue_mask_threshold_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_raw = tissue_mask(rgb, args.seed_white_threshold, args.seed_saturation_threshold)
    seed, seed_meta = remove_small_components(seed_raw, args.seed_min_area)
    print(f"seed_meta={seed_meta}", flush=True)
    distance = ndi.distance_transform_edt(~seed)

    white_thresholds = parse_ints(args.white_thresholds)
    saturation_thresholds = parse_ints(args.saturation_thresholds)
    distance_values = parse_ints(args.distance_px)

    by_class_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    visual_candidates: list[tuple[float, str, str, np.ndarray]] = []

    seed_params = {
        "seed_white_threshold": args.seed_white_threshold,
        "seed_saturation_threshold": args.seed_saturation_threshold,
        "seed_min_area": args.seed_min_area,
    }
    for white_threshold in white_thresholds:
        for saturation_threshold in saturation_thresholds:
            candidate = tissue_mask(rgb, white_threshold, saturation_threshold)
            for distance_px in distance_values:
                if distance_px <= 0:
                    mask = candidate
                    mask_name = f"raw_w{white_threshold}_s{saturation_threshold}"
                else:
                    mask = candidate & (distance <= distance_px)
                    mask_name = f"w{white_threshold}_s{saturation_threshold}_dist{distance_px}"
                params = {
                    "white_threshold": white_threshold,
                    "saturation_threshold": saturation_threshold,
                    "distance_px": distance_px,
                    **seed_params,
                }
                rows_for_mask = coverage_rows(mask, gt, class_names, mask_name, params, totals)
                by_class_rows.extend(rows_for_mask)
                background = class_fraction(rows_for_mask, "background")
                fat = class_fraction(rows_for_mask, "fat")
                submucosa = class_fraction(rows_for_mask, "submucosa_serosa")
                tissue_classes = [item for item in rows_for_mask if item["class_name"] != "background"]
                mean_non_background = float(np.mean([float(item["covered_fraction"]) for item in tissue_classes]))
                score = fat - background
                summary_rows.append(
                    {
                        "mask_name": mask_name,
                        "white_threshold": white_threshold,
                        "saturation_threshold": saturation_threshold,
                        "distance_px": distance_px,
                        "background_fraction": background,
                        "fat_fraction": fat,
                        "submucosa_serosa_fraction": submucosa,
                        "mean_non_background_fraction": mean_non_background,
                        "score_fat_minus_background": score,
                    }
                )
                if fat >= 0.85:
                    subtitle = f"bg={background:.3f} fat={fat:.3f} sub={submucosa:.3f}"
                    visual_candidates.append((score, mask_name, subtitle, mask.copy()))
            print(f"scanned white={white_threshold} saturation={saturation_threshold}", flush=True)

    write_csv(
        output_dir / "threshold_distance_by_class.csv",
        by_class_rows,
        [
            "mask_name",
            "white_threshold",
            "saturation_threshold",
            "distance_px",
            "seed_white_threshold",
            "seed_saturation_threshold",
            "seed_min_area",
            "class_id",
            "class_name",
            "pixels",
            "covered_pixels",
            "covered_fraction",
        ],
    )
    write_csv(
        output_dir / "threshold_distance_summary.csv",
        summary_rows,
        [
            "mask_name",
            "white_threshold",
            "saturation_threshold",
            "distance_px",
            "background_fraction",
            "fat_fraction",
            "submucosa_serosa_fraction",
            "mean_non_background_fraction",
            "score_fat_minus_background",
        ],
    )

    top_rows = sorted(
        [row for row in summary_rows if float(row["fat_fraction"]) >= 0.9],
        key=lambda row: (float(row["background_fraction"]), -float(row["submucosa_serosa_fraction"])),
    )[: args.top_k]
    write_csv(
        output_dir / "top_fat90_candidates.csv",
        top_rows,
        [
            "mask_name",
            "white_threshold",
            "saturation_threshold",
            "distance_px",
            "background_fraction",
            "fat_fraction",
            "submucosa_serosa_fraction",
            "mean_non_background_fraction",
            "score_fat_minus_background",
        ],
    )
    if top_rows:
        top_names = {row["mask_name"] for row in top_rows}
        panels = [(name, subtitle, mask) for _, name, subtitle, mask in visual_candidates if name in top_names]
        write_visual(output_dir / "top_fat90_candidates.png", rgb, panels[: args.top_k], args.max_dim)

    meta = {
        "image_id": args.image_id,
        "rgb_shape": list(rgb.shape),
        "analysis_max_dim": args.analysis_max_dim,
        "seed": {
            "white_threshold": args.seed_white_threshold,
            "saturation_threshold": args.seed_saturation_threshold,
            "min_area": args.seed_min_area,
            **seed_meta,
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    for row in top_rows[:5]:
        print(
            f"{row['mask_name']} bg={float(row['background_fraction']):.4f} "
            f"fat={float(row['fat_fraction']):.4f} sub={float(row['submucosa_serosa_fraction']):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
