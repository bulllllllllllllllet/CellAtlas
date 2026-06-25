from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image
from torchvision import transforms

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.patch_geometry import rectangle_majority_label
from new_inference_stream import inference as inference_base


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def process_slide_patches(
    row: dict[str, str],
    output_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    white_threshold: int,
    blank_ratio: float,
) -> dict:
    slide = openslide.OpenSlide(row["he_path"])
    width, height = slide.level_dimensions[0]
    gt_mask = np.load(output_root / "masks" / f"{row['image_id']}_gt_mask.npy", mmap_mode="r")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    records = []
    tensors = []
    feature_parts = []
    total_grid_patches = 0
    skipped_blank_patches = 0
    started = time.time()

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode():
            feature_parts.append(model(torch.stack(tensors).to(device)).cpu().numpy())
        tensors.clear()

    try:
        for y0 in range(0, height, stride):
            for x0 in range(0, width, stride):
                total_grid_patches += 1
                x1, y1 = min(x0 + patch_size, width), min(y0 + patch_size, height)
                image = slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0)).convert("RGB")
                array = np.asarray(image)
                white_ratio = float(np.mean(array.mean(axis=2) >= white_threshold))
                if white_ratio >= blank_ratio:
                    skipped_blank_patches += 1
                    continue
                label, purity, pixels = rectangle_majority_label(
                    gt_mask, (width, height), (x0, y0, x1, y1)
                )
                records.append(
                    {
                        "patch_index": len(records),
                        "x0_original": x0,
                        "y0_original": y0,
                        "x1_original": x1,
                        "y1_original": y1,
                        "center_x_original": (x0 + x1) / 2,
                        "center_y_original": (y0 + y1) / 2,
                        "gt_tissue_label": label,
                        "gt_label_purity": purity,
                        "gt_pixels": pixels,
                        "white_ratio": white_ratio,
                    }
                )
                tensors.append(preprocess(image))
                if len(tensors) >= batch_size:
                    flush()
    finally:
        slide.close()
    flush()
    features = (
        np.concatenate(feature_parts).astype(np.float32, copy=False)
        if feature_parts
        else np.empty((0, 768), dtype=np.float32)
    )
    passed = len(records) == len(features) and len(records) > 0 and np.isfinite(features).all()
    output_dir = output_root / "patches" / row["image_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not passed:
        raise RuntimeError(
            f"patch validation failed before write: records={len(records)} features={features.shape}"
        )
    raw_path = output_dir / "raw.npy"
    temporary_raw = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with open(temporary_raw, "wb") as file:
        np.save(file, features)
    temporary_raw.replace(raw_path)
    patches_path = output_dir / "patches.csv"
    temporary_patches = patches_path.with_suffix(patches_path.suffix + ".tmp")
    with open(temporary_patches, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_patches.replace(patches_path)
    valid_class_counts = {
        str(class_id): sum(
            int(record["gt_tissue_label"]) == class_id
            and float(record["gt_label_purity"]) >= 0.7
            for record in records
        )
        for class_id in range(12)
    }
    report = {
        "image_id": row["image_id"],
        "patch_size": patch_size,
        "stride": stride,
        "white_threshold": white_threshold,
        "blank_ratio": blank_ratio,
        "total_grid_patches": total_grid_patches,
        "skipped_blank_patches": skipped_blank_patches,
        "patches": len(records),
        "feature_shape": list(features.shape),
        "valid_class_patch_counts": valid_class_counts,
        "purity_definition": "majority_class_pixels / all_patch_gt_pixels",
        "elapsed_seconds": time.time() - started,
        "passed": bool(passed),
    }
    validation_path = output_dir / "validation.json"
    temporary_validation = validation_path.with_suffix(validation_path.suffix + ".tmp")
    with open(temporary_validation, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary_validation.replace(validation_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dense-grid GDPH CTransPath patch tokens.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv")
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--white_threshold", type=int, default=240)
    parser.add_argument("--blank_ratio", type=float, default=0.995)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = _read_csv(args.manifest)
    if args.image_id:
        requested = set(args.image_id)
        rows = [row for row in rows if row["image_id"] in requested]
    if not args.force:
        pending_rows = []
        for row in rows:
            validation_path = Path(args.output_root) / "patches" / row["image_id"] / "validation.json"
            if validation_path.exists():
                with open(validation_path, "r", encoding="utf-8") as file:
                    validation = json.load(file)
                if (
                    validation.get("passed")
                    and validation.get("patch_size") == args.patch_size
                    and validation.get("stride") == args.stride
                    and validation.get("white_threshold") == args.white_threshold
                    and validation.get("blank_ratio") == args.blank_ratio
                ):
                    print(f"[{row['image_id']}] patch inference already complete", flush=True)
                    continue
            pending_rows.append(row)
        rows = pending_rows
    if not rows:
        print("all requested patch slides are already complete", flush=True)
        return
    device = torch.device(args.device)
    model = inference_base.load_ctranspath(args.ctranspath_checkpoint, device)
    for row in rows:
        report = process_slide_patches(
            row,
            Path(args.output_root),
            model,
            device,
            args.patch_size,
            args.stride,
            args.batch_size,
            args.white_threshold,
            args.blank_ratio,
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
