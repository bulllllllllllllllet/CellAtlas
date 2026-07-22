"""Cached multi-scale token dataset for Phase-4 training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TokenCacheDataset(Dataset):
    def __init__(self, cache_index: str | Path, label_index: str | Path, split: str):
        cache = pd.read_parquet(cache_index)
        cache = cache[cache["split"] == split].reset_index(drop=True)
        labels = pd.read_parquet(label_index)
        labels = labels[labels["split"] == split].set_index("patch_id")
        missing = [pid for pid in cache["patch_id"] if pid not in labels.index]
        if missing:
            raise ValueError(f"label index missing {len(missing)} patches for split={split}")
        self.rows = cache.to_dict("records")
        self.labels = labels.loc[cache["patch_id"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        lab = self.labels[index]
        z = np.load(row["shard_path"])
        labels = np.load(lab["label_path"]).astype(np.int64)
        return {
            "patch_id": row["patch_id"],
            "wsi_id": row["wsi_id"],
            "sampling_group": row["sampling_group"],
            "fine_tokens": torch.from_numpy(z["fine_tokens"].astype(np.float32)),
            "middle_tokens": torch.from_numpy(z["middle_tokens"].astype(np.float32)),
            "coarse_tokens": torch.from_numpy(z["coarse_tokens"].astype(np.float32)),
            "fine_active": torch.from_numpy(z["fine_active"].astype(np.bool_)),
            "fine_middle_edge_index": torch.from_numpy(z["fine_middle_edge_index"].astype(np.int64)),
            "fine_middle_edge_weight": torch.from_numpy(z["fine_middle_edge_weight"].astype(np.float32)),
            "middle_coarse_edge_index": torch.from_numpy(z["middle_coarse_edge_index"].astype(np.int64)),
            "middle_coarse_edge_weight": torch.from_numpy(z["middle_coarse_edge_weight"].astype(np.float32)),
            "labels": torch.from_numpy(labels),
        }


def collate_token_cache(items: list[dict]) -> dict:
    out = {
        "patch_id": [x["patch_id"] for x in items],
        "wsi_id": [x["wsi_id"] for x in items],
        "sampling_group": [x["sampling_group"] for x in items],
    }
    keys = [
        "fine_tokens",
        "middle_tokens",
        "coarse_tokens",
        "fine_active",
        "fine_middle_edge_index",
        "fine_middle_edge_weight",
        "middle_coarse_edge_index",
        "middle_coarse_edge_weight",
        "labels",
    ]
    for key in keys:
        out[key] = torch.stack([x[key] for x in items])
    return out


class GroupBalancedSampler(torch.utils.data.Sampler[int]):
    """Balanced sampler over sampling_group field on TokenCacheDataset rows."""

    def __init__(self, dataset: TokenCacheDataset, ratios: dict[str, float], seed: int, rank: int = 0, world_size: int = 1, epoch_size: int | None = None):
        self.groups = {
            name: np.asarray([i for i, r in enumerate(dataset.rows) if r["sampling_group"] == name], dtype=np.int64)
            for name in ratios
        }
        absent = [name for name, ids in self.groups.items() if len(ids) == 0]
        if absent:
            raise ValueError(f"empty sampling groups: {absent}")
        if not np.isclose(sum(ratios.values()), 1.0):
            raise ValueError("ratios must sum to 1")
        self.ratios = ratios
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        requested = epoch_size if epoch_size is not None else len(dataset)
        self.global_length = (requested // world_size) * world_size
        if not self.global_length:
            raise ValueError("epoch size smaller than world size")
        self.length = self.global_length // world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        picks = []
        for name, ratio in self.ratios.items():
            n = int(round(self.global_length * ratio))
            ids = self.groups[name]
            choices = rng.choice(ids, size=n, replace=len(ids) < n)
            picks.append(choices)
        order = np.concatenate(picks)
        rng.shuffle(order)
        order = order[: self.global_length]
        return iter(order[self.rank :: self.world_size].tolist())

    def __len__(self) -> int:
        return self.length
