from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from benchmarks.v4.phase_1_multiscale.src.dataset import SegmentationDataset


def canonicalize_slic(labels: np.ndarray, num_slots: int = 64) -> np.ndarray:
    """Match SLIC centroids one-to-one to fixed spatial slots."""
    grid_size=int(round(num_slots**0.5))
    if grid_size*grid_size != num_slots: raise ValueError(f"num_slots must be a square number, got {num_slots}")
    ids=np.unique(labels); centroids=[]
    for region_id in ids:
        yy,xx=np.nonzero(labels==region_id); centroids.append((float(yy.mean())/labels.shape[0],float(xx.mean())/labels.shape[1],int(region_id)))
    if len(centroids)>num_slots: raise ValueError(f"SLIC patch has {len(centroids)} regions but only {num_slots} slots")
    anchors=np.asarray([((row+.5)/grid_size,(col+.5)/grid_size) for row in range(grid_size) for col in range(grid_size)])
    points=np.asarray([(y,x) for y,x,_ in centroids]); cost=((points[:,None]-anchors[None])**2).sum(-1); region_rows,slot_cols=linear_sum_assignment(cost)
    lookup={centroids[row][2]:int(slot) for row,slot in zip(region_rows,slot_cols,strict=True)}
    out=np.empty(labels.shape,dtype=np.uint8)
    for old,new in lookup.items(): out[labels==old]=new
    return out


class RegionDataset(SegmentationDataset):
    def __init__(self, patch_index: str | Path, slic_index: str | Path, config: dict, split: str, augment: bool = False):
        super().__init__(patch_index, config, split, augment=False)
        self.region_augment=augment; self.num_regions=int(config["model"]["num_regions"])
        refs=pd.read_parquet(slic_index).query("split == @split").set_index("patch_id")
        missing=[r["patch_id"] for r in self.rows if r["patch_id"] not in refs.index]
        if missing: raise ValueError(f"SLIC index misses {len(missing)} patches in split={split}")
        self.refs=refs.loc[[r["patch_id"] for r in self.rows]].to_dict("records"); self._open=OrderedDict()

    def _slic(self, ref: dict) -> np.ndarray:
        path=Path(ref["shard_path"])
        if path.suffix != ".npy": raise ValueError(f"Phase 2 training requires mmap .npy SLIC shards, got: {path}")
        data=self._open.get(path)
        if data is None:
            data=np.load(path,mmap_mode="r"); self._open[path]=data
            if len(self._open) > 16:
                _,old=self._open.popitem(last=False)
                if getattr(old,"_mmap",None) is not None: old._mmap.close()
        else: self._open.move_to_end(path)
        return np.array(data[int(ref["offset"])],dtype=np.uint8,copy=True)

    def __getitem__(self, index: int) -> dict:
        item=super().__getitem__(index); slic=self._slic(self.refs[index])
        if slic.shape != tuple(item["mask"].shape): raise RuntimeError(f"SLIC shape mismatch for {item['patch_id']}: {slic.shape}")
        if self.region_augment:
            k=int(torch.randint(0,4,()).item()); item["image"]=torch.rot90(item["image"],k,(1,2)); item["mask"]=torch.rot90(item["mask"],k,(0,1)); slic=np.rot90(slic,k).copy()
            if torch.rand(()) < .5: item["image"]=item["image"].flip(2); item["mask"]=item["mask"].flip(1); slic=np.fliplr(slic).copy()
            if torch.rand(()) < .5: item["image"]=item["image"].flip(1); item["mask"]=item["mask"].flip(0); slic=np.flipud(slic).copy()
        item["slic"] = torch.from_numpy(canonicalize_slic(slic,self.num_regions))
        return item
