"""Physical multi-scale region geometry and sparse parent-child edges."""
from __future__ import annotations

import numpy as np


SCALE_SPECS = {
    "10x": {"field_level0_mult": 1, "prefix": ""},
    "5x": {"field_level0_mult": 2, "prefix": "_5x"},
    "2p5x": {"field_level0_mult": 4, "prefix": "_2p5x"},
}


def scale_level0_box(row: dict, scale: str) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) of a scale crop in level-0 coordinates."""
    if scale not in SCALE_SPECS:
        raise ValueError(f"unsupported scale: {scale}")
    if scale == "10x":
        return (
            int(row["x_level0"]),
            int(row["y_level0"]),
            int(row["width_level0"]),
            int(row["height_level0"]),
        )
    prefix = SCALE_SPECS[scale]["prefix"]
    return (
        int(row[f"x{prefix}_level0"]),
        int(row[f"y{prefix}_level0"]),
        int(row[f"width{prefix}_level0"]),
        int(row[f"height{prefix}_level0"]),
    )


def assignment_centroids_areas(
    assignment: np.ndarray,
    x0: int,
    y0: int,
    width0: int,
    height0: int,
    mass_threshold: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Map soft assignment mass to physical centroids/areas in level-0 coords.

    assignment: [K, H, W] probabilities over the resized patch grid.
    """
    if assignment.ndim != 3:
        raise ValueError(f"assignment must be [K,H,W], got {assignment.shape}")
    k, h, w = assignment.shape
    mass = assignment.reshape(k, -1).sum(axis=1).astype(np.float64)
    active = mass > mass_threshold
    # pixel centers in patch-normalized [0,1], then map to level-0
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    level0_x = x0 + grid_x * width0
    level0_y = y0 + grid_y * height0
    cx = np.zeros(k, dtype=np.float64)
    cy = np.zeros(k, dtype=np.float64)
    for slot in range(k):
        if not active[slot]:
            continue
        m = assignment[slot].astype(np.float64)
        total = m.sum()
        cx[slot] = float((m * level0_x).sum() / total)
        cy[slot] = float((m * level0_y).sum() / total)
    # area in level-0 pixels: mass * (level0_pixel_area / patch_pixels)
    pixel_area = (width0 * height0) / float(h * w)
    area = mass * pixel_area
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "area": area,
        "mass": mass,
        "active": active,
    }


def parent_child_edges(
    child_assignment: np.ndarray,
    child_box: tuple[int, int, int, int],
    parent_assignment: np.ndarray,
    parent_box: tuple[int, int, int, int],
    top_k: int = 4,
    mass_threshold: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Build sparse top-k parent edges from soft assignment overlap in physical space.

    Both assignments are defined on their own resized patch grids. Overlap is
    estimated by projecting child pixel centers into the parent patch and
    bilinearly sampling parent assignment mass.
    """
    if child_assignment.ndim != 3 or parent_assignment.ndim != 3:
        raise ValueError("assignments must be [K,H,W]")
    ck, ch, cw = child_assignment.shape
    pk, ph, pw = parent_assignment.shape
    cx0, cy0, cw0, ch0 = child_box
    px0, py0, pw0, ph0 = parent_box
    child_mass = child_assignment.reshape(ck, -1).sum(axis=1)
    child_active = child_mass > mass_threshold
    parent_mass = parent_assignment.reshape(pk, -1).sum(axis=1)
    parent_active = parent_mass > mass_threshold

    # child pixel centers in level-0
    ys = (np.arange(ch, dtype=np.float64) + 0.5) / ch
    xs = (np.arange(cw, dtype=np.float64) + 0.5) / cw
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    level0_x = cx0 + grid_x * cw0
    level0_y = cy0 + grid_y * ch0
    # map to parent normalized coords in [-1,1] for bilinear sample
    parent_u = (level0_x - px0) / max(pw0, 1) * 2.0 - 1.0
    parent_v = (level0_y - py0) / max(ph0, 1) * 2.0 - 1.0
    # bilinear sample parent assignment at each child pixel
    # convert to parent pixel continuous coords
    parent_xf = (parent_u + 1.0) * 0.5 * (pw - 1)
    parent_yf = (parent_v + 1.0) * 0.5 * (ph - 1)
    # clamp for border sampling
    parent_xf = np.clip(parent_xf, 0.0, pw - 1.000001)
    parent_yf = np.clip(parent_yf, 0.0, ph - 1.000001)
    x0 = np.floor(parent_xf).astype(np.int64)
    y0 = np.floor(parent_yf).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, pw - 1)
    y1 = np.clip(y0 + 1, 0, ph - 1)
    wx = parent_xf - x0
    wy = parent_yf - y0
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w10 = wx * (1.0 - wy)
    w11 = wx * wy
    # parent_sampled[k, y, x]
    parent_sampled = (
        parent_assignment[:, y0, x0] * w00
        + parent_assignment[:, y1, x0] * w01
        + parent_assignment[:, y0, x1] * w10
        + parent_assignment[:, y1, x1] * w11
    )
    # overlap[child_k, parent_k] = sum_pixels child_mass * parent_prob
    flat_child = child_assignment.reshape(ck, -1)
    flat_parent = parent_sampled.reshape(pk, -1)
    overlap = flat_child @ flat_parent.T  # [Ck, Pk]
    # zero inactive parents
    overlap[:, ~parent_active] = 0.0
    overlap[~child_active, :] = 0.0

    k_eff = min(top_k, pk)
    edge_index = np.full((ck, k_eff), -1, dtype=np.int16)
    edge_weight = np.zeros((ck, k_eff), dtype=np.float32)
    weight_sum = np.zeros(ck, dtype=np.float64)
    isolated = np.zeros(ck, dtype=bool)
    for child in range(ck):
        if not child_active[child]:
            continue
        scores = overlap[child]
        total = float(scores.sum())
        if total <= 1e-12:
            isolated[child] = True
            continue
        top = np.argpartition(scores, -k_eff)[-k_eff:]
        top = top[np.argsort(-scores[top])]
        # Keep only positive-mass parents among the top-k, then renormalize so
        # each active child's sparse edges sum to 1 for message passing.
        top_scores = scores[top]
        positive = top_scores > 0
        if not positive.any():
            isolated[child] = True
            continue
        top = top[positive]
        top_scores = top_scores[positive]
        weights = top_scores / top_scores.sum()
        edge_index[child, : len(top)] = top.astype(np.int16)
        edge_weight[child, : len(weights)] = weights.astype(np.float32)
        weight_sum[child] = float(weights.sum())
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "weight_sum": weight_sum,
        "overlap": overlap.astype(np.float32),
        "child_active": child_active,
        "parent_active": parent_active,
        "isolated_active_children": isolated & child_active,
    }


def edge_invariants(edge_payload: dict, atol: float = 1e-3) -> dict:
    child_active = edge_payload["child_active"]
    isolated = edge_payload["isolated_active_children"]
    weight_sum = edge_payload["weight_sum"]
    active_ids = np.where(child_active)[0]
    if active_ids.size == 0:
        return {
            "active_children": 0,
            "isolated_active_children": 0,
            "weight_sum_mean": None,
            "weight_sum_max_abs_error": None,
            "weight_sum_ok": True,
            "no_isolated_active": True,
            "passed": True,
        }
    errors = np.abs(weight_sum[active_ids] - 1.0)
    # isolated children cannot satisfy weight-sum==1
    non_isolated = active_ids[~isolated[active_ids]]
    if non_isolated.size:
        errors = np.abs(weight_sum[non_isolated] - 1.0)
        max_err = float(errors.max())
        mean_sum = float(weight_sum[non_isolated].mean())
        weight_ok = bool(max_err <= atol)
    else:
        max_err = None
        mean_sum = None
        weight_ok = False
    report = {
        "active_children": int(child_active.sum()),
        "isolated_active_children": int(isolated.sum()),
        "weight_sum_mean": mean_sum,
        "weight_sum_max_abs_error": max_err,
        "weight_sum_ok": weight_ok,
        "no_isolated_active": bool(isolated.sum() == 0),
        "passed": bool(weight_ok and isolated.sum() == 0),
    }
    return report
