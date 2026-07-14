# 作用：生成阶段 A small/medium/large 多尺度 superpixel 的 HE 与 GT 叠加可视化检查图。

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
LEGACY_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2")
DATA_MANIFEST = V3_ROOT / "pret_superpixel" / "data_manifest_v3.csv"
PALETTE_JSON = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/gdph_tissue_palette.json")
REPORT_DIR = Path(__file__).resolve().parent
SCALES = ["small", "medium", "large"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def target_size(shape: tuple[int, int], max_dim: int) -> tuple[int, int]:
    height, width = shape
    if max(height, width) <= max_dim:
        return width, height
    scale = max_dim / max(height, width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_array_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(array).astype(np.int32), mode="I").resize(size, Image.NEAREST), dtype=np.int32)


def resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    rgb = np.load(path, mmap_mode="r")
    return np.asarray(Image.fromarray(np.asarray(rgb)).resize(size, Image.BILINEAR), dtype=np.uint8)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    valid = labels >= 0
    boundary = np.zeros(labels.shape, dtype=bool)
    diff_x = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    diff_y = valid[1:, :] & valid[:-1, :] & (labels[1:, :] != labels[:-1, :])
    boundary[:, 1:] |= diff_x
    boundary[:, :-1] |= diff_x
    boundary[1:, :] |= diff_y
    boundary[:-1, :] |= diff_y
    return boundary


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            y_src0 = max(0, -dy)
            y_src1 = mask.shape[0] - max(0, dy)
            x_src0 = max(0, -dx)
            x_src1 = mask.shape[1] - max(0, dx)
            y_dst0 = max(0, dy)
            y_dst1 = mask.shape[0] - max(0, -dy)
            x_dst0 = max(0, dx)
            x_dst1 = mask.shape[1] - max(0, -dx)
            shifted[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
            out |= shifted
    return out


def overlay_boundary(
    base: np.ndarray,
    labels: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    invalid_color: tuple[int, int, int] | None = None,
    invalid_mask: np.ndarray | None = None,
) -> Image.Image:
    image = np.asarray(base).copy()
    if invalid_color is not None:
        mask = labels < 0 if invalid_mask is None else invalid_mask
        image[mask] = invalid_color
    boundary = dilate(boundary_mask(labels), radius)
    image[boundary] = color
    return Image.fromarray(image, mode="RGB")


def load_palette(path: Path) -> np.ndarray:
    colors = np.zeros((256, 3), dtype=np.uint8)
    colors[:] = [35, 35, 35]
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("classes", []):
        colors[int(item["id"])] = item["rgb"]
    colors[255] = [0, 0, 0]
    return colors


def gt_color_image(gt: np.ndarray, palette: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.int32)
    if gt.ndim == 3:
        gt = rgb_to_class_mask(gt.astype(np.uint8), palette)
    clipped = np.where((gt >= 0) & (gt < len(palette)), gt, 255)
    return palette[clipped].astype(np.uint8)


def rgb_to_class_mask(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    output = np.full(rgb.shape[:2], 255, dtype=np.int32)
    for class_id, color in enumerate(palette[:255]):
        matches = np.all(rgb == color.reshape(1, 1, 3), axis=2)
        output[matches] = class_id
    return output


def load_gt_mask(row: dict[str, str], size: tuple[int, int], output_root: Path) -> np.ndarray:
    cached = LEGACY_ROOT / "masks" / f"{row['image_id']}_gt_mask.npy"
    if cached.exists():
        return resize_array_nearest(np.load(cached, mmap_mode="r"), size)
    return np.asarray(Image.open(row["gt_mask_path"]).resize(size, Image.NEAREST), dtype=np.int32)


def add_header(panel: Image.Image, title: str, subtitle: str) -> Image.Image:
    header_h = 56
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
        x = (index % columns) * width
        y = (index // columns) * height
        grid.paste(panel, (x, y))
    return grid


def render_image(row: dict[str, str], output_root: Path, visual_root: Path, max_dim: int, boundary_radius: int, palette: np.ndarray) -> dict[str, object]:
    image_id = row["image_id"]
    image_root = output_root / "pret_superpixel" / "multiscale_tokens" / image_id
    medium_dir = image_root / "medium"
    medium_segments = np.load(medium_dir / "superpixels.npy", mmap_mode="r")
    size = target_size(tuple(medium_segments.shape), max_dim)
    rgb = resize_rgb(medium_dir / "he_10x_rgb.npy", size)
    gt = load_gt_mask(row, size, output_root)
    gt_rgb = gt_color_image(gt, palette)

    panels: list[Image.Image] = []
    records: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for scale in SCALES:
        scale_dir = image_root / scale
        segments = resize_array_nearest(np.load(scale_dir / "superpixels.npy", mmap_mode="r"), size)
        validation = json.loads((scale_dir / "validation.json").read_text(encoding="utf-8"))
        subtitle = f"segments={validation['segment_count']} edges={validation['adjacency_edges']}"
        panels.append(add_header(overlay_boundary(rgb, segments, (255, 25, 25), boundary_radius), f"{scale} | HE + boundary", subtitle))
        panels.append(
            add_header(
                overlay_boundary(
                    gt_rgb,
                    segments,
                    (255, 255, 255),
                    boundary_radius,
                    invalid_color=(0, 0, 0),
                    invalid_mask=(segments < 0) & (gt != 2),
                ),
                f"{scale} | GT + boundary",
                subtitle + " | black=uncovered",
            )
        )
        records.append(
            {
                "image_id": image_id,
                "scale": scale,
                "segment_count": validation["segment_count"],
                "adjacency_edges": validation["adjacency_edges"],
                "base_token_shape": "x".join(map(str, validation["base_token_shape"])),
                "enhanced_token_shape": "x".join(map(str, validation["enhanced_token_shape"])),
            }
        )
        for class_id in sorted(int(value) for value in np.unique(gt)):
            mask = gt == class_id
            pixels = int(mask.sum())
            if pixels == 0:
                continue
            invalid = int(((segments < 0) & mask).sum())
            invalid_rows.append(
                {
                    "image_id": image_id,
                    "scale": scale,
                    "class_id": class_id,
                    "pixels": pixels,
                    "invalid_pixels": invalid,
                    "invalid_fraction": invalid / pixels,
                }
            )

    # Reorder to two rows: HE small/medium/large, then GT small/medium/large.
    ordered = [panels[i] for i in [0, 2, 4, 1, 3, 5]]
    grid = make_grid(ordered, columns=3)
    out_dir = visual_root / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiscale_superpixel_gt_check.png"
    grid.save(out_path)
    return {
        "image_id": image_id,
        "image_file": str(out_path),
        "render_width": grid.width,
        "render_height": grid.height,
        "scale_rows": records,
        "invalid_rows": invalid_rows,
    }


def write_report(path: Path, rows: list[dict[str, object]], index_path: Path, visual_root: Path) -> None:
    lines = [
        "# Phase A 多尺度可视化检查报告",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- visual_root: `{visual_root}`",
        f"- index_csv: `{index_path}`",
        f"- rendered_images: {len(rows)}",
        "",
        "## 使用方法",
        "",
        "- 打开每个 `multiscale_superpixel_gt_check.png`。",
        "- 第一行是 HE + superpixel boundary，第二行是 GT mask + superpixel boundary。",
        "- GT 行里的黑色区域表示该尺度 superpixel 没有覆盖到的像素，即 `superpixels == -1`。",
        "- 横向依次为 small、medium、large，应该看到边界从细到粗变化。",
        "- 如果 large 大量跨越明显不同 GT 类别，说明当前 hierarchical merge 需要改进。",
        "- 如果某类组织内部有大量黑点/黑块，说明 tissue mask 或 superpixel 生成覆盖不足。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render phase A multiscale superpixel visual checks.")
    parser.add_argument("--data_manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=V3_ROOT)
    parser.add_argument("--visual_root", type=Path, default=V3_ROOT / "pret_superpixel" / "visualizations" / "phase_a_multiscale_check")
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--max_images", type=int, default=0, help="0 renders all images.")
    parser.add_argument("--max_dim", type=int, default=2048)
    parser.add_argument("--boundary_radius", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.data_manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError("no images selected")
    palette = load_palette(PALETTE_JSON)

    index_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        result = render_image(row, args.output_root, args.visual_root, args.max_dim, args.boundary_radius, palette)
        index_rows.append({key: result[key] for key in ["image_id", "image_file", "render_width", "render_height"]})
        scale_rows.extend(result["scale_rows"])
        invalid_rows.extend(result["invalid_rows"])
        print(f"rendered {index}/{len(rows)} image_id={row['image_id']}", flush=True)

    write_csv(args.visual_root / "visual_index.csv", index_rows, ["image_id", "image_file", "render_width", "render_height"])
    write_csv(args.visual_root / "scale_stats.csv", scale_rows, ["image_id", "scale", "segment_count", "adjacency_edges", "base_token_shape", "enhanced_token_shape"])
    write_csv(args.visual_root / "invalid_fraction_by_class.csv", invalid_rows, ["image_id", "scale", "class_id", "pixels", "invalid_pixels", "invalid_fraction"])
    payload = {
        "passed": len(index_rows) == len(rows) and all(Path(str(row["image_file"])).exists() for row in index_rows),
        "rendered_images": len(index_rows),
        "visual_root": str(args.visual_root),
        "index_csv": str(args.visual_root / "visual_index.csv"),
    }
    (args.visual_root / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(REPORT_DIR / "visual_check_report.md", index_rows, args.visual_root / "visual_index.csv", args.visual_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise RuntimeError("visual check validation failed")


if __name__ == "__main__":
    main()
