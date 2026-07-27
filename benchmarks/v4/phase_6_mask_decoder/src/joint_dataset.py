"""Online HE/GT/cell data joined to audited cached prompt episodes."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch, read_he_patch
from benchmarks.v4.phase_2_region_encoder.src.dataset import canonicalize_slic
from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells, encode_xcell_features
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor, read_scale_image
from benchmarks.v4.phase_4_cross_scale.src.geometry import scale_level0_box
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset, collate_prompt_episodes


class RoutedCellFeatures:
    def __init__(self, routing: str | Path, patch_index: str | Path, split: str, max_open_shards: int = 16):
        patches = pd.read_parquet(patch_index).query("split == @split").reset_index(drop=True)
        route = pd.read_parquet(routing).sort_values("source_index").reset_index(drop=True)
        if len(route) != len(patches):
            raise ValueError(f"routing/patch count mismatch for {split}: {len(route)} != {len(patches)}")
        if not np.array_equal(route["source_index"].to_numpy(), np.arange(len(route))):
            raise ValueError("feature routing source_index is not contiguous")
        self.refs = {str(patch_id): ref for patch_id, ref in zip(patches["patch_id"], route.to_dict("records"), strict=True)}
        self.cache: OrderedDict[Path, pd.DataFrame] = OrderedDict(); self.max_open_shards = int(max_open_shards)

    def get(self, patch_id: str) -> tuple[np.ndarray, int]:
        ref = self.refs.get(str(patch_id))
        if ref is None:
            raise KeyError(f"cell routing misses patch {patch_id}")
        path = Path(ref["feature_shard_path"]); table = self.cache.get(path)
        if table is None:
            table = pd.read_parquet(path, columns=["patch_id", "cells", "reg_features", "total_cell_count"])
            self.cache[path] = table
            if len(self.cache) > self.max_open_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        row = table.iloc[int(ref["row_offset"])]
        if str(row.patch_id) != str(patch_id):
            raise RuntimeError(f"cell feature route mismatch: {row.patch_id} != {patch_id}")
        cells = np.stack(row.cells).astype(np.float32) if len(row.cells) else np.empty((0, 4), np.float32)
        reg = np.stack(row.reg_features).astype(np.float32) if len(row.reg_features) else np.empty((0, 64), np.float32)
        return encode_xcell_features(cells, reg), int(row.total_cell_count)


class RoutedSLICLabels:
    """Random-access canonical SLIC labels for the fixed-region A8 control."""
    def __init__(self, slic_index: str | Path, patch_ids: set[str], num_regions: int, max_open_shards: int = 16):
        refs = pd.read_parquet(slic_index)
        refs = refs[refs["patch_id"].astype(str).isin(patch_ids)].set_index("patch_id")
        missing = patch_ids - set(refs.index.astype(str))
        if missing:
            raise ValueError(f"SLIC index misses {len(missing)} prompt-episode patches")
        self.refs = {str(key): value for key, value in refs.to_dict("index").items()}
        self.num_regions = int(num_regions); self.max_open_shards = int(max_open_shards)
        self.cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, patch_id: str, expected_shape: tuple[int, int]) -> np.ndarray:
        ref = self.refs.get(str(patch_id))
        if ref is None:
            raise KeyError(f"SLIC routing misses patch {patch_id}")
        path = Path(ref["shard_path"]); data = self.cache.get(path)
        if data is None:
            if path.suffix != ".npy":
                raise ValueError(f"fixed SLIC requires mmap .npy shards, got {path}")
            data = np.load(path, mmap_mode="r"); self.cache[path] = data
            if len(self.cache) > self.max_open_shards:
                _, old = self.cache.popitem(last=False)
                if getattr(old, "_mmap", None) is not None:
                    old._mmap.close()
        else:
            self.cache.move_to_end(path)
        labels = np.array(data[int(ref["offset"])], dtype=np.uint8, copy=True)
        if labels.shape != expected_shape:
            raise RuntimeError(f"SLIC/image shape mismatch for {patch_id}: {labels.shape} != {expected_shape}")
        return canonicalize_slic(labels, self.num_regions)


class JointPixelEpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: PromptEpisodeDataset,
        patch_index: str | Path,
        cell_routing: str | Path,
        class_map: list[dict],
        ignore_index: int,
        max_cells: int,
        include_multiscale: bool = False,
        include_parent_context: bool = False,
        slic_index: str | Path | None = None,
        slic_num_regions: int | None = None,
    ):
        self.episodes = episodes; self.ignore = int(ignore_index); self.class_map = class_map; self.max_cells = int(max_cells)
        self.include_multiscale = bool(include_multiscale)
        self.include_parent_context = bool(include_parent_context)
        # Routing manifests do not universally carry split; infer it from the episode dataset's patch rows.
        patch_frame = pd.read_parquet(patch_index)
        episode_ids = set(map(str, episodes.patch_ids))
        candidates = patch_frame[patch_frame["patch_id"].astype(str).isin(episode_ids)]
        splits = candidates["split"].unique().tolist()
        if len(splits) != 1:
            raise ValueError(f"joint dataset expected one split, found {splits}")
        self.split = str(splits[0]); self.rows = candidates.set_index("patch_id").to_dict("index")
        if len(self.rows) != len(set(episodes.patch_ids)):
            raise ValueError("patch index does not cover every prompt episode patch")
        self.cells = RoutedCellFeatures(cell_routing, patch_index, self.split)
        if (slic_index is None) != (slic_num_regions is None):
            raise ValueError("fixed SLIC requires both slic_index and slic_num_regions")
        self.slic = RoutedSLICLabels(slic_index, set(map(str, episodes.patch_ids)), int(slic_num_regions)) if slic_index is not None else None

    def __len__(self) -> int:
        return len(self.episodes)

    def set_epoch(self, epoch: int) -> None:
        self.episodes.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        item = self.episodes[index]; row = self.rows[str(item["patch_id"])]
        image = read_he_patch(
            Path(row["wsi_path"]), int(row["x_level0"]), int(row["y_level0"]),
            int(row["width_level0"]), int(row["height_level0"]), int(row["width_10x"]),
        )
        mask = decode_gt_patch(
            Path(row["gt_path"]), int(row["x_10x"]), int(row["y_10x"]),
            int(row["width_10x"]), int(row["height_10x"]), self.class_map, self.ignore,
        )
        if image.shape[:2] != mask.shape:
            raise RuntimeError(f"HE/GT mismatch for {item['patch_id']}: {image.shape} vs {mask.shape}")
        cells, total = self.cells.get(str(item["patch_id"]))
        multiscale = {}
        if self.include_multiscale:
            # Every crop is rendered to the fine-model input resolution, but
            # each one covers its own nested level-0 field of view.
            multiscale = {
                "image_5x": he_to_tensor(read_scale_image(row, "5x")),
                "image_2p5x": he_to_tensor(read_scale_image(row, "2p5x")),
                "box_10x": scale_level0_box(row, "10x"),
                "box_5x": scale_level0_box(row, "5x"),
                "box_2p5x": scale_level0_box(row, "2p5x"),
            }
        # Historical J10 evaluation remains readable, but new Phase-4 runs use
        # include_multiscale and never consult these cached parent tokens.
        if self.include_parent_context:
            with np.load(self.episodes.shard_paths[index]) as archive:
                keys = ("middle_tokens", "coarse_tokens", "fine_middle_edge_index", "fine_middle_edge_weight", "middle_coarse_edge_index", "middle_coarse_edge_weight")
                missing = [key for key in keys if key not in archive]
                if missing:
                    raise ValueError(f"historical parent cache for {item['patch_id']} misses {missing}")
                multiscale.update({
                    "middle_tokens": torch.from_numpy(archive["middle_tokens"].astype(np.float32)),
                    "coarse_tokens": torch.from_numpy(archive["coarse_tokens"].astype(np.float32)),
                    "fine_middle_edge_index": torch.from_numpy(archive["fine_middle_edge_index"].astype(np.int64)),
                    "fine_middle_edge_weight": torch.from_numpy(archive["fine_middle_edge_weight"].astype(np.float32)),
                    "middle_coarse_edge_index": torch.from_numpy(archive["middle_coarse_edge_index"].astype(np.int64)),
                    "middle_coarse_edge_weight": torch.from_numpy(archive["middle_coarse_edge_weight"].astype(np.float32)),
                })
        item.update({
            "episode_index": torch.tensor(index, dtype=torch.long),
            "image": he_to_tensor(image), "pixel_gt": torch.from_numpy(mask.astype(np.int64)),
            "cells_raw": cells, "total_cell_count_raw": total,
            **multiscale,
        })
        if self.slic is not None:
            item["fixed_slic"] = torch.from_numpy(self.slic.get(str(item["patch_id"]), mask.shape).astype(np.int64))
        return item


def collate_joint_pixel_episodes(items: list[dict], max_cells: int) -> dict:
    batch = collate_prompt_episodes(items)
    packed = collate_cells(
        [(str(item["patch_id"]), item["cells_raw"], int(item["total_cell_count_raw"])) for item in items],
        max_cells=int(max_cells),
    )
    batch.update(packed)
    if "image_5x" in items[0]:
        batch.update({
            "image_5x": torch.stack([item["image_5x"] for item in items]),
            "image_2p5x": torch.stack([item["image_2p5x"] for item in items]),
            # Keep physical boxes on CPU as Python values; they are consumed
            # only while constructing detached sparse graph topology.
            "box_10x": [item["box_10x"] for item in items],
            "box_5x": [item["box_5x"] for item in items],
            "box_2p5x": [item["box_2p5x"] for item in items],
        })
    if "middle_tokens" in items[0]:
        batch.update({name: torch.stack([item[name] for item in items]) for name in (
            "middle_tokens", "coarse_tokens", "fine_middle_edge_index", "fine_middle_edge_weight",
            "middle_coarse_edge_index", "middle_coarse_edge_weight",
        )})
    if "fixed_slic" in items[0]:
        batch["fixed_slic"] = torch.stack([item["fixed_slic"] for item in items])
    return batch
