"""Dataset/collate for batched multi-scale token cache export."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells, encode_xcell_features
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor, read_scale_image
from benchmarks.v4.phase_4_cross_scale.src.geometry import scale_level0_box


class MultiscaleCacheDataset(Dataset):
    """Prefetch 10x/5x/2.5x HE tensors and hybrid cell features for one rank shard."""

    def __init__(self, items: list[dict], max_cells: int, max_open_shards: int = 16):
        self.items = items
        self.max_cells = max_cells
        self.max_open_shards = max_open_shards
        self._feature_cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _features(self, route_row: dict, patch_id: str) -> tuple[np.ndarray, int]:
        path = Path(route_row["feature_shard_path"])
        table = self._feature_cache.get(path)
        if table is None:
            table = pd.read_parquet(path, columns=["patch_id", "cells", "reg_features", "total_cell_count"])
            self._feature_cache[path] = table
            if len(self._feature_cache) > self.max_open_shards:
                self._feature_cache.popitem(last=False)
        else:
            self._feature_cache.move_to_end(path)
        row = table.iloc[int(route_row["row_offset"])]
        if str(row.patch_id) != patch_id:
            raise RuntimeError(f"feature route mismatch: {row.patch_id} != {patch_id}")
        cells = np.stack(row.cells).astype(np.float32) if len(row.cells) else np.empty((0, 4), np.float32)
        reg = np.stack(row.reg_features).astype(np.float32) if len(row.reg_features) else np.empty((0, 64), np.float32)
        return encode_xcell_features(cells, reg), int(row.total_cell_count)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        row = item["row"]
        patch_id = row["patch_id"]
        features, total = self._features(item["route"], patch_id)
        images = {}
        boxes = {}
        for scale in ("10x", "5x", "2p5x"):
            rgb = read_scale_image(row, scale)
            images[scale] = he_to_tensor(rgb)
            boxes[scale] = scale_level0_box(row, scale)
        return {
            "patch_id": patch_id,
            "wsi_id": row["wsi_id"],
            "split": item["split"],
            "sampling_group": row.get("sampling_group"),
            "image_10x": images["10x"],
            "image_5x": images["5x"],
            "image_2p5x": images["2p5x"],
            "box_10x": boxes["10x"],
            "box_5x": boxes["5x"],
            "box_2p5x": boxes["2p5x"],
            "cells": features,
            "total_cell_count": total,
        }


def collate_multiscale_cache(items: list[dict], max_cells: int) -> dict:
    cell_pack = collate_cells(
        [(x["patch_id"], x["cells"], x["total_cell_count"]) for x in items],
        max_cells,
    )
    return {
        "patch_id": [x["patch_id"] for x in items],
        "wsi_id": [x["wsi_id"] for x in items],
        "split": [x["split"] for x in items],
        "sampling_group": [x["sampling_group"] for x in items],
        "image_10x": torch.stack([x["image_10x"] for x in items]),
        "image_5x": torch.stack([x["image_5x"] for x in items]),
        "image_2p5x": torch.stack([x["image_2p5x"] for x in items]),
        "box_10x": [x["box_10x"] for x in items],
        "box_5x": [x["box_5x"] for x in items],
        "box_2p5x": [x["box_2p5x"] for x in items],
        **cell_pack,
    }


def balance_by_wsi(items: list[dict], rank: int, world: int) -> list[dict]:
    """Assign whole WSI groups with greedy load balancing on patch counts.

    Keeps slide affinity (fewer unique WSI opens per rank) while preventing one
    rank from receiving far more patches than others.
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["row"]["wsi_id"], []).append(item)
    # largest groups first for better bin packing, stable by wsi_id
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    bins: list[list[dict]] = [[] for _ in range(world)]
    loads = [0 for _ in range(world)]
    for _, group in ordered:
        target = min(range(world), key=lambda i: (loads[i], i))
        bins[target].extend(group)
        loads[target] += len(group)
    shard = bins[rank]
    shard.sort(key=lambda x: (x["row"]["wsi_id"], x["row"]["patch_id"]))
    return shard
