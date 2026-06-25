from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmarks.tissue_seg.io import cell_centroids_and_gt_labels, load_or_create_gt_mask, read_mapping_csv
from benchmarks.tissue_seg.palette import IGNORE_LABEL, load_palette


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-cell tissue labels by sampling GT class masks at cell centroids."
    )
    parser.add_argument("--mapping_csv", required=True, help="CSV with image_id,gt_path,masks_path,...")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--palette_json", default=None)
    parser.add_argument("--color_tolerance", type=float, default=18.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gt_mask_dir = output_dir / "gt_masks"
    cell_label_dir = output_dir / "cell_labels"
    cell_label_dir.mkdir(parents=True, exist_ok=True)

    palette = load_palette(args.palette_json)
    rows = read_mapping_csv(args.mapping_csv)

    manifest_path = output_dir / "prepared_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as manifest_file:
        fieldnames = [
            "image_id",
            "gt_mask_path",
            "cell_labels_path",
            "masks_path",
            "reg_path",
            "proj_path",
            "raw_path",
            "he_path",
        ]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            masks = np.load(row.masks_path, mmap_mode="r")
            gt_mask_path = gt_mask_dir / f"{row.image_id}_gt_mask.npy"
            gt_mask = load_or_create_gt_mask(
                row.gt_path,
                gt_mask_path,
                palette=palette,
                tolerance=args.color_tolerance,
            )
            cell_ids, centroids, gt_labels = cell_centroids_and_gt_labels(masks, gt_mask, IGNORE_LABEL)

            cell_labels_path = cell_label_dir / f"{row.image_id}_cell_labels.csv"
            with open(cell_labels_path, "w", encoding="utf-8", newline="") as f:
                label_writer = csv.DictWriter(
                    f,
                    fieldnames=["cell_index", "cell_id", "centroid_x", "centroid_y", "gt_label"],
                )
                label_writer.writeheader()
                for idx, (cell_id, centroid, gt_label) in enumerate(
                    zip(cell_ids, centroids, gt_labels, strict=False)
                ):
                    label_writer.writerow(
                        {
                            "cell_index": idx,
                            "cell_id": int(cell_id),
                            "centroid_x": int(centroid[0]),
                            "centroid_y": int(centroid[1]),
                            "gt_label": int(gt_label),
                        }
                    )

            writer.writerow(
                {
                    "image_id": row.image_id,
                    "gt_mask_path": str(gt_mask_path),
                    "cell_labels_path": str(cell_labels_path),
                    "masks_path": str(row.masks_path),
                    "reg_path": str(row.reg_path or ""),
                    "proj_path": str(row.proj_path or ""),
                    "raw_path": str(row.raw_path or ""),
                    "he_path": str(row.he_path or ""),
                }
            )
            valid = int(np.sum(gt_labels != IGNORE_LABEL))
            print(f"[prepare] {row.image_id}: cells={len(gt_labels)} valid={valid}")

    print(f"[prepare] wrote {manifest_path}")


if __name__ == "__main__":
    main()
