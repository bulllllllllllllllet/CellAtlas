from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _sorted_labels_by_centroid(region: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return np.empty(0, dtype=np.int64)
    labels = region[ys, xs].astype(np.int64, copy=False)
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    sum_y = np.bincount(labels, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(labels, weights=xs, minlength=max_label + 1)
    valid = np.where(counts > 0)[0]
    valid = valid[valid > 0]
    cy = sum_y[valid] / counts[valid]
    cx = sum_x[valid] / counts[valid]
    order = np.lexsort((cx, cy))
    return valid[order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel tiled inference masks so label ids align with concatenated token order."
    )
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--features_path", required=True, help="Feature .npy used only to validate token count")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tile_size", type=int, default=2048)
    parser.add_argument("--max_cells", type=int, default=255)
    args = parser.parse_args()

    masks = np.load(args.masks_path, mmap_mode="r")
    features = np.load(args.features_path, mmap_mode="r")
    h, w = masks.shape
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int32,
        shape=masks.shape,
    )
    aligned[:] = 0

    next_label = 1
    for y in range(0, h, args.tile_size):
        for x in range(0, w, args.tile_size):
            y1 = min(y + args.tile_size, h)
            x1 = min(x + args.tile_size, w)
            region = masks[y:y1, x:x1]
            labels = _sorted_labels_by_centroid(region)[: args.max_cells]
            if labels.size == 0:
                continue

            out_region = aligned[y:y1, x:x1]
            for source_label in labels:
                out_region[region == source_label] = next_label
                next_label += 1

    token_count = int(features.shape[0])
    aligned_count = next_label - 1
    if aligned_count != token_count:
        raise SystemExit(
            f"Aligned mask has {aligned_count} labels, but feature array has {token_count} rows."
        )
    aligned.flush()
    print(f"{output_path} labels={aligned_count} shape={h}x{w}")


if __name__ == "__main__":
    main()
