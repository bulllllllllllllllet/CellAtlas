from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from benchmarks.v4.phase_2_region_encoder.src.dataset import RegionDataset
from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells,encode_xcell_features


class CellRegionDataset(Dataset):
    """Phase-2 image/SLIC samples plus lazily routed hybrid XCell features."""
    def __init__(self, patch_index, slic_index, routing, config, split: str, max_open_shards: int = 16):
        self.region=RegionDataset(patch_index,slic_index,config,split,augment=False)
        route=pd.read_parquet(routing).sort_values("source_index").reset_index(drop=True)
        if len(route)!=len(self.region): raise ValueError(f"routing/patch count mismatch for {split}: {len(route)} != {len(self.region)}")
        if not np.array_equal(route.source_index.to_numpy(),np.arange(len(route))): raise ValueError("routing source_index is not contiguous")
        self.route=route.to_dict("records"); self.cache=OrderedDict(); self.max_open_shards=max_open_shards

    def __len__(self): return len(self.region)

    def _feature(self, ref: dict, patch_id: str) -> tuple[np.ndarray,int]:
        path=Path(ref["feature_shard_path"]); table=self.cache.get(path)
        if table is None:
            table=pd.read_parquet(path,columns=["patch_id","cells","reg_features","total_cell_count"]); self.cache[path]=table
            if len(self.cache)>self.max_open_shards: self.cache.popitem(last=False)
        else: self.cache.move_to_end(path)
        row=table.iloc[int(ref["row_offset"])]
        if row.patch_id!=patch_id: raise RuntimeError(f"feature route mismatch: {row.patch_id} != {patch_id}")
        cells=np.stack(row.cells).astype(np.float32) if len(row.cells) else np.empty((0,4),np.float32)
        reg=np.stack(row.reg_features).astype(np.float32) if len(row.reg_features) else np.empty((0,64),np.float32)
        try: features=encode_xcell_features(cells,reg)
        except ValueError as exc: raise RuntimeError(f"{patch_id}: {exc}") from exc
        return features,int(row.total_cell_count)

    def __getitem__(self,index):
        item=self.region[index]; cells,total=self._feature(self.route[index],item["patch_id"]); item["cells"]=cells; item["total_cell_count"]=total; return item


def collate_cell_region(items: list[dict], max_cells: int) -> dict:
    cells=collate_cells([(x["patch_id"],x["cells"],x["total_cell_count"]) for x in items],max_cells)
    return {"image":torch.stack([x["image"] for x in items]),"mask":torch.stack([x["mask"] for x in items]),"slic":torch.stack([x["slic"] for x in items]),"patch_id":[x["patch_id"] for x in items],**cells}
