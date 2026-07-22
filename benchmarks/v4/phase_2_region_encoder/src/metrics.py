from __future__ import annotations

import numpy as np


def hard_region_metrics(region: np.ndarray, target: np.ndarray, ignore_index: int, num_classes: int) -> dict[str, float]:
    """Validation metrics from a hard region-id image."""
    valid = target != ignore_index
    purities=[]; oracle=np.full(target.shape, ignore_index, dtype=np.int64)
    for rid in np.unique(region[valid]):
        inside=(region == rid) & valid; labels=target[inside]; counts=np.bincount(labels, minlength=num_classes)
        purities.append(float(counts.max() / counts.sum()))
        oracle[inside]=int(counts.argmax())
    gt_edge=np.zeros_like(valid); gt_edge[1:] |= (target[1:] != target[:-1]) & valid[1:] & valid[:-1]; gt_edge[:, 1:] |= (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    reg_edge=np.zeros_like(valid); reg_edge[1:] |= region[1:] != region[:-1]; reg_edge[:, 1:] |= region[:, 1:] != region[:, :-1]
    tp=(gt_edge & reg_edge & valid).sum(); precision=tp / max((reg_edge & valid).sum(), 1); recall=tp / max((gt_edge & valid).sum(), 1)
    dice=[]
    for c in range(num_classes):
        g=(target == c) & valid; p=(oracle == c) & valid
        if g.any(): dice.append(2 * (g & p).sum() / max(g.sum() + p.sum(), 1))
    return {"region_purity": float(np.mean(purities)) if purities else 0.0, "boundary_f1": float(2 * precision * recall / max(precision + recall, 1e-8)), "oracle_region_dice": float(np.mean(dice)) if dice else 0.0, "active_regions": float(len(np.unique(region[valid])))}


def region_metrics(assignment: np.ndarray, target: np.ndarray, ignore_index: int, num_classes: int) -> dict[str, float]:
    """Compatibility wrapper for soft assignments."""
    return hard_region_metrics(assignment.argmax(0),target,ignore_index,num_classes)
