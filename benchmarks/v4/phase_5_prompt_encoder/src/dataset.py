"""Cached fine-token episodic dataset for Phase-5 prompt training."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .prompts import PROMPT_SIZE_SPECS, centroid_knn_adjacency, sample_connected_region_set


SIZE_NAMES = tuple(PROMPT_SIZE_SPECS)


class PromptEpisodeDataset(Dataset):
    """Generate one deterministic feasible target-vs-rest episode per patch."""

    def __init__(
        self,
        cache_index: str | Path,
        label_index: str | Path,
        patch_index: str | Path,
        split: str,
        seed: int,
        size_probabilities: dict[str, float],
        target_class_ids: tuple[int, ...],
        ignore_index: int = 255,
        centroid_knn: int = 4,
        eligibility_index: str | Path | None = None,
    ):
        cache = pd.read_parquet(cache_index)
        cache = cache[cache["split"] == split].reset_index(drop=True)
        labels = pd.read_parquet(label_index)
        labels = labels[labels["split"] == split].set_index("patch_id")
        patches = pd.read_parquet(patch_index).set_index("patch_id")
        missing_labels = [pid for pid in cache["patch_id"] if pid not in labels.index]
        missing_patches = [pid for pid in cache["patch_id"] if pid not in patches.index]
        if missing_labels or missing_patches:
            raise ValueError(
                f"missing inputs labels={len(missing_labels)} patches={len(missing_patches)}"
            )
        if set(size_probabilities) != set(SIZE_NAMES):
            raise ValueError(f"size probabilities must define {SIZE_NAMES}")
        total_probability = float(sum(size_probabilities.values()))
        if not np.isclose(total_probability, 1.0):
            raise ValueError("size probabilities must sum to 1")
        if eligibility_index is not None:
            eligible = pd.read_parquet(eligibility_index)
            eligible = eligible[eligible["split"] == split].reset_index(drop=True)
            if eligible.empty:
                raise ValueError(f"eligibility index has no rows for split={split}")
            cache = eligible.merge(
                cache,
                on=["patch_id", "wsi_id", "split", "sampling_group"],
                how="left",
                validate="many_to_one",
            )
            if cache["shard_path"].isna().any():
                raise ValueError("eligibility index contains patches absent from token cache")
        patch_ids = cache["patch_id"]
        patch_geometry = patches.loc[patch_ids]
        self.patch_ids = patch_ids.astype(str).to_numpy()
        self.wsi_ids = cache["wsi_id"].astype(str).to_numpy()
        self.sampling_groups = cache["sampling_group"].astype(str).to_numpy()
        self.shard_paths = cache["shard_path"].astype(str).to_numpy()
        self.label_paths = labels.loc[patch_ids, "label_path"].astype(str).to_numpy()
        self.patch_boxes = patch_geometry[["x_level0", "y_level0", "width_level0", "height_level0"]].to_numpy(dtype=np.float64)
        self.prompt_sizes = cache["prompt_size"].astype(str).to_numpy() if "prompt_size" in cache else None
        self.target_classes = cache["target_class"].to_numpy(dtype=np.int64) if "target_class" in cache else None
        self.positive_slot_sets = cache["positive_slots"].to_numpy() if "positive_slots" in cache else None
        self.seed = int(seed)
        self.epoch = 0
        self.size_probabilities = {str(k): float(v) for k, v in size_probabilities.items()}
        self.target_class_ids = tuple(map(int, target_class_ids))
        self.ignore_index = int(ignore_index)
        self.centroid_knn = int(centroid_knn)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)

    def __getitem__(self, index: int) -> dict:
        patch_id = self.patch_ids[index]
        with np.load(self.shard_paths[index]) as archive:
            fine = archive["fine_tokens"].astype(np.float32)
            active = archive["fine_active"].astype(bool)
            cx = archive["fine_centroid_x"].astype(np.float32)
            cy = archive["fine_centroid_y"].astype(np.float32)
            area = archive["fine_area"].astype(np.float32)
        labels = np.load(self.label_paths[index]).astype(np.int64)
        valid = active & (labels != self.ignore_index)
        if not valid.any():
            raise RuntimeError(f"no valid fine regions for {patch_id}")
        x0, y0, width, height = self.patch_boxes[index]
        xy = np.stack([(cx - x0) / width, (cy - y0) / height], axis=1).astype(np.float32)
        area_fraction = area / max(float(area[active].sum()), 1e-6)
        adjacency = centroid_knn_adjacency(cx, cy, active, self.centroid_knn)
        rng = self._rng(index)

        if self.prompt_sizes is not None:
            size_name = str(self.prompt_sizes[index])
            target = int(self.target_classes[index])
            raw_slots = self.positive_slot_sets[index]
            if isinstance(raw_slots, str):
                raw_slots = json.loads(raw_slots)
            positive_slots = np.asarray(raw_slots, dtype=np.int64)
            if not np.all(valid[positive_slots] & (labels[positive_slots] == target)):
                raise RuntimeError(f"stale eligibility row for {patch_id}")
        else:
            feasible: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in SIZE_NAMES}
            present = sorted(set(map(int, labels[valid])) & set(self.target_class_ids))
            for candidate_size, candidate_spec in PROMPT_SIZE_SPECS.items():
                for candidate_target in present:
                    eligible = valid & (labels == candidate_target)
                    negative_count = int((valid & (labels != candidate_target)).sum())
                    if negative_count < int(candidate_spec["negative_slots"]):
                        continue
                    try:
                        slots = sample_connected_region_set(
                            adjacency,
                            eligible,
                            np.maximum(np.rint(area), 1).astype(np.int64),
                            rng,
                            min_slots=int(candidate_spec["min_slots"]),
                            max_slots=int(candidate_spec["max_slots"]),
                            min_fraction=float(candidate_spec["min_fraction"]),
                            max_fraction=float(candidate_spec["max_fraction"]),
                            patch_pixels=max(int(np.rint(area[active].sum())), 1),
                        )
                    except ValueError:
                        continue
                    feasible[candidate_size].append((candidate_target, slots))
            available_sizes = [name for name in SIZE_NAMES if feasible[name]]
            if not available_sizes:
                raise RuntimeError(f"no feasible prompt episode for {patch_id}")
            weights = np.asarray([self.size_probabilities[name] for name in available_sizes], dtype=np.float64)
            weights /= weights.sum()
            size_name = str(rng.choice(available_sizes, p=weights))
            target, positive_slots = feasible[size_name][int(rng.integers(0, len(feasible[size_name])))]
        spec = PROMPT_SIZE_SPECS[size_name]
        negative_pool = np.flatnonzero(valid & (labels != target))
        center = xy[positive_slots].mean(0)
        distance = ((xy[negative_pool] - center) ** 2).sum(1)
        negative_slots = negative_pool[np.argsort(distance)[: int(spec["negative_slots"])]].astype(np.int64)

        max_positive = max(int(item["max_slots"]) for item in PROMPT_SIZE_SPECS.values())
        max_negative = max(int(item["negative_slots"]) for item in PROMPT_SIZE_SPECS.values())
        positive_tokens = np.zeros((max_positive, fine.shape[1]), dtype=np.float32)
        negative_tokens = np.zeros((max_negative, fine.shape[1]), dtype=np.float32)
        positive_xy = np.zeros((max_positive, 2), dtype=np.float32)
        negative_xy = np.zeros((max_negative, 2), dtype=np.float32)
        positive_mask = np.zeros(max_positive, dtype=bool)
        negative_mask = np.zeros(max_negative, dtype=bool)
        positive_slot_indices = np.full(max_positive, -1, dtype=np.int64)
        negative_slot_indices = np.full(max_negative, -1, dtype=np.int64)
        positive_tokens[: len(positive_slots)] = fine[positive_slots]
        negative_tokens[: len(negative_slots)] = fine[negative_slots]
        positive_xy[: len(positive_slots)] = xy[positive_slots]
        negative_xy[: len(negative_slots)] = xy[negative_slots]
        positive_mask[: len(positive_slots)] = True
        negative_mask[: len(negative_slots)] = True
        positive_slot_indices[: len(positive_slots)] = positive_slots
        negative_slot_indices[: len(negative_slots)] = negative_slots
        prompted_regions = np.zeros(len(labels), dtype=bool)
        prompted_regions[positive_slots] = True
        negative_prompted_regions = np.zeros(len(labels), dtype=bool)
        negative_prompted_regions[negative_slots] = True
        all_prompted_regions = prompted_regions | negative_prompted_regions
        binary_target = np.full(len(labels), self.ignore_index, dtype=np.int64)
        binary_target[valid] = (labels[valid] == target).astype(np.int64)
        return {
            "patch_id": patch_id,
            "wsi_id": self.wsi_ids[index],
            "sampling_group": self.sampling_groups[index],
            "prompt_size": size_name,
            "prompt_size_id": torch.tensor(SIZE_NAMES.index(size_name), dtype=torch.long),
            "target_class": torch.tensor(target, dtype=torch.long),
            "fine_tokens": torch.from_numpy(fine),
            "fine_active": torch.from_numpy(active),
            "region_xy": torch.from_numpy(xy),
            "region_area": torch.from_numpy(area_fraction),
            "positive_tokens": torch.from_numpy(positive_tokens),
            "negative_tokens": torch.from_numpy(negative_tokens),
            "positive_xy": torch.from_numpy(positive_xy),
            "negative_xy": torch.from_numpy(negative_xy),
            "positive_mask": torch.from_numpy(positive_mask),
            "negative_mask": torch.from_numpy(negative_mask),
            "positive_slot_indices": torch.from_numpy(positive_slot_indices),
            "negative_slot_indices": torch.from_numpy(negative_slot_indices),
            "prompted_regions": torch.from_numpy(prompted_regions),
            "negative_prompted_regions": torch.from_numpy(negative_prompted_regions),
            "all_prompted_regions": torch.from_numpy(all_prompted_regions),
            "binary_target": torch.from_numpy(binary_target),
        }


def collate_prompt_episodes(items: list[dict]) -> dict:
    out = {
        "patch_id": [item["patch_id"] for item in items],
        "wsi_id": [item["wsi_id"] for item in items],
        "sampling_group": [item["sampling_group"] for item in items],
        "prompt_size": [item["prompt_size"] for item in items],
    }
    for key in items[0]:
        if key not in out and torch.is_tensor(items[0][key]):
            out[key] = torch.stack([item[key] for item in items])
    return out


class EpisodeBalancedSampler(torch.utils.data.Sampler[int]):
    """Sample prompt episodes over class, size, and source sampling group.

    Each non-empty joint bucket receives probability proportional to the
    product of its requested marginals. With a complete joint support this
    gives uniform target classes and the exact requested size/group ratios in
    expectation. Empty joint buckets are explicit in ``empty_buckets``.
    """

    def __init__(
        self,
        dataset: PromptEpisodeDataset,
        size_probabilities: dict[str, float],
        group_probabilities: dict[str, float],
        class_ids: tuple[int, ...],
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        epoch_size: int | None = None,
    ):
        if not np.isclose(sum(size_probabilities.values()), 1.0):
            raise ValueError("size probabilities must sum to 1")
        if not np.isclose(sum(group_probabilities.values()), 1.0):
            raise ValueError("group probabilities must sum to 1")
        self.categories = []
        self.buckets = []
        self.empty_buckets = []
        self.empty_group_buckets = []
        weights = []
        available: dict[tuple[int, str, str], list[int]] = {}
        if dataset.target_classes is None or dataset.prompt_sizes is None:
            raise ValueError("EpisodeBalancedSampler requires an eligibility-backed dataset")
        for index, (class_id, size, group) in enumerate(zip(dataset.target_classes, dataset.prompt_sizes, dataset.sampling_groups)):
            key = (int(class_id), str(size), str(group))
            available.setdefault(key, []).append(index)
        for class_id in class_ids:
            for size, size_probability in size_probabilities.items():
                category = (int(class_id), str(size))
                group_buckets = []
                for group, group_probability in group_probabilities.items():
                    group_category = (*category, str(group))
                    ids = np.asarray(available.get(group_category, []), dtype=np.int64)
                    if not len(ids):
                        self.empty_group_buckets.append(group_category)
                        continue
                    group_buckets.append((ids, float(group_probability)))
                if not group_buckets:
                    self.empty_buckets.append(category)
                    continue
                self.categories.append(category); self.buckets.append(group_buckets)
                weights.append(float(size_probability) / len(class_ids))
        if not self.buckets:
            raise ValueError("no non-empty episode buckets")
        self.weights = np.asarray(weights, dtype=np.float64); self.weights /= self.weights.sum()
        self.seed = int(seed); self.rank = int(rank); self.world_size = int(world_size); self.epoch = 0
        requested = len(dataset) if epoch_size is None else int(epoch_size)
        self.global_length = (requested // self.world_size) * self.world_size
        if not self.global_length:
            raise ValueError("epoch size smaller than world size")
        self.length = self.global_length // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        category_ids = rng.choice(len(self.buckets), size=self.global_length, p=self.weights)
        picks = np.empty(self.global_length, dtype=np.int64)
        for category_id, group_buckets in enumerate(self.buckets):
            positions = np.flatnonzero(category_ids == category_id)
            if len(positions):
                group_weights = np.asarray([item[1] for item in group_buckets], dtype=np.float64)
                group_weights /= group_weights.sum()
                selected_groups = rng.choice(len(group_buckets), size=len(positions), p=group_weights)
                for group_id, (bucket, _) in enumerate(group_buckets):
                    local = positions[selected_groups == group_id]
                    if len(local):
                        picks[local] = rng.choice(bucket, size=len(local), replace=len(bucket) < len(local))
        return iter(picks[self.rank :: self.world_size].tolist())

    def __len__(self) -> int:
        return self.length
