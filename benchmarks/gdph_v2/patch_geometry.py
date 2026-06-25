from __future__ import annotations

import numpy as np


def rectangle_majority_label(
    gt_mask: np.ndarray,
    original_size: tuple[int, int],
    rectangle_original: tuple[int, int, int, int],
    num_classes: int = 12,
) -> tuple[int, float, int]:
    gt_height, gt_width = gt_mask.shape
    original_width, original_height = original_size
    x0, y0, x1, y1 = rectangle_original
    gx0 = max(0, int(np.floor(x0 * gt_width / original_width)))
    gy0 = max(0, int(np.floor(y0 * gt_height / original_height)))
    gx1 = min(gt_width, max(gx0 + 1, int(np.ceil(x1 * gt_width / original_width))))
    gy1 = min(gt_height, max(gy0 + 1, int(np.ceil(y1 * gt_height / original_height))))
    values = np.asarray(gt_mask[gy0:gy1, gx0:gx1]).reshape(-1)
    valid = values[(values >= 0) & (values < num_classes)]
    if valid.size == 0:
        return 255, 0.0, 0
    counts = np.bincount(valid.astype(np.int64), minlength=num_classes)
    label = int(np.argmax(counts))
    return label, float(counts[label] / values.size), int(values.size)
