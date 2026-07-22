#!/usr/bin/env python3
"""Match GDPH tissue masks with nucleus annotations and visualize alignment.

Example (run from repository root):
    python benchmarks/v4/data_preprocess/match_gdph_tissue_nuclei.py

The CSV paths are stored relative to ``--common-root`` so that the shared NFS
prefix is not duplicated in every row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import pyvips
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "Visualization requires pyvips. Run this script with the project's "
        "conda environment: conda run -n aligner python ..."
    ) from exc


COMMON_ROOT = Path(
    "/nfs-medical/vipa-medical/2024.8.12广东省人民医院移动硬盘数据/"
    "结直肠癌/结直肠癌组织分割和细胞核分割"
)
DEFAULT_TISSUE_DIR = COMMON_ROOT / "tissue_seg@10x/GDPH-tissue-segmentation/GDPH-tissue-segmentation"
DEFAULT_NUCLEI_DIR = COMMON_ROOT / "DP500-COAD_READ-GDPH_P01"
DEFAULT_CSV = Path("benchmarks/v4/data_preprocess/gdph_tissue_nuclei_pairs.csv")
DEFAULT_VIS_DIR = Path("/nfs-medical3/zyh/v4/dataset/visualization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-root", type=Path, default=COMMON_ROOT)
    parser.add_argument("--tissue-dir", type=Path, default=DEFAULT_TISSUE_DIR)
    parser.add_argument("--nuclei-dir", type=Path, default=DEFAULT_NUCLEI_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--visualization-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--num-visualizations", type=int, default=20)
    return parser.parse_args()


def relative(path: Path, common_root: Path) -> str:
    """Return a POSIX path relative to common_root, falling back to absolute."""
    try:
        return path.relative_to(common_root).as_posix()
    except ValueError:
        return str(path)


def find_matches(args: argparse.Namespace) -> tuple[list[dict[str, Path]], list[str]]:
    records: list[dict[str, Path]] = []
    missing: list[str] = []
    for tissue_path in sorted(args.tissue_dir.glob("*.png")):
        stem = tissue_path.stem
        candidates = []
        for part in range(1, 6):
            folder = args.nuclei_dir / f"part{part}"
            candidate = {
                "tissue": tissue_path,
                "class": folder / f"{stem}-class.tiff",
                "instance": folder / f"{stem}-instance.tiff",
                "image": folder / f"{stem}.tiff",
            }
            if all(path.is_file() for path in candidate.values()):
                candidates.append(candidate)
        if len(candidates) == 1:
            records.append(candidates[0])
        elif not candidates:
            missing.append(f"{tissue_path.name}: no complete nucleus annotation triplet")
        else:
            parts = ", ".join(item["image"].parent.name for item in candidates)
            missing.append(f"{tissue_path.name}: duplicate matches in {parts}")
    return records, missing


def resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def tissue_overlay(image: Image.Image, tissue: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = np.asarray(tissue.convert("L")) > 0
    if mask.shape != base.shape[:2]:
        mask = np.asarray(tissue.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 0
    base[mask] = base[mask] * 0.55 + np.array([0, 255, 80]) * 0.45
    return Image.fromarray(base.astype(np.uint8))


def class_overlay(image: Image.Image, class_map: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    labels = np.asarray(class_map)
    if labels.ndim == 3:
        labels = labels[..., 0]
    if labels.shape != base.shape[:2]:
        labels = np.asarray(Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST))
    palette = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
                        [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                        [240, 50, 230], [210, 245, 60]], dtype=np.float32)
    color = palette[labels.astype(np.int64) % len(palette)]
    foreground = labels > 0
    base[foreground] = base[foreground] * 0.35 + color[foreground] * 0.65
    return Image.fromarray(base.astype(np.uint8))


def instance_boundary_overlay(image: Image.Image, instance_map: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    labels = np.asarray(instance_map)
    if labels.ndim == 3:
        labels = labels[..., 0]
    if labels.shape != base.shape[:2]:
        labels = np.asarray(Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST))
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary &= labels > 0
    base[boundary] = [255, 0, 255]
    return Image.fromarray(base)


def add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 26), "white")
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((6, 5), title, fill="black", font=ImageFont.load_default())
    return canvas


def read_center_preview(path: Path, crop_size: int, output_size: int = 768) -> Image.Image:
    """Read a central crop using libvips, without materializing a whole slide."""
    source = pyvips.Image.new_from_file(str(path), access="sequential")
    side = min(crop_size, source.width, source.height)
    left = (source.width - side) // 2
    top = (source.height - side) // 2
    preview = source.crop(left, top, side, side).resize(output_size / side)
    array = np.ndarray(
        buffer=preview.write_to_memory(),
        dtype=np.uint8,
        shape=(preview.height, preview.width, preview.bands),
    )
    if preview.bands == 1:
        return Image.fromarray(array[..., 0])
    if preview.bands >= 3:
        return Image.fromarray(array[..., :3])
    return Image.fromarray(array.squeeze())


def visualize(record: dict[str, Path], output_path: Path) -> None:
    # The tissue PNG is at 10x; the three TIFF files are at 20x.  Read a
    # matching central region and downsample TIFFs by 2x to the PNG scale.
    tissue = read_center_preview(record["tissue"], crop_size=3072)
    original = read_center_preview(record["image"], crop_size=6144)
    classes = read_center_preview(record["class"], crop_size=6144)
    instances = read_center_preview(record["instance"], crop_size=6144)
    # Keep output manageable while retaining enough detail to inspect registration.
    width, height = original.size
    target = (width, height)
    panels = [
        add_title(original.convert("RGB"), "original TIFF (10x)"),
        add_title(tissue.convert("RGB"), "tissue segmentation PNG (10x)"),
        add_title(tissue_overlay(original, tissue), "tissue mask on original"),
        add_title(class_overlay(original, classes), "nucleus class overlay"),
        add_title(instance_boundary_overlay(original, instances), "instance boundaries"),
    ]
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, (height + 26) * rows), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * (height + 26)))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    records, issues = find_matches(args)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["分割图路径", "细胞核分类", "实例分割结果", "原图"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "分割图路径": relative(record["tissue"], args.common_root),
                "细胞核分类": relative(record["class"], args.common_root),
                "实例分割结果": relative(record["instance"], args.common_root),
                "原图": relative(record["image"], args.common_root),
            })
    args.visualization_dir.mkdir(parents=True, exist_ok=True)
    for record in records[:args.num_visualizations]:
        visualize(record, args.visualization_dir / f"{record['image'].stem}_alignment.png")
    print(f"Matched {len(records)} records; CSV: {args.csv_path}")
    print(f"Wrote {min(len(records), args.num_visualizations)} visualizations to {args.visualization_dir}")
    if issues:
        print(f"Skipped {len(issues)} tissue masks:")
        print("\n".join(issues[:20]))


if __name__ == "__main__":
    main()
