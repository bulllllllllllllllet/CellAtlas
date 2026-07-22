from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .data import decode_gt_patch, read_he_patch


class SegmentationDataset(Dataset):
    def __init__(self, index_path: str | Path, config: dict, split: str, augment: bool = False):
        self.rows = pd.read_parquet(index_path).query("split == @split").to_dict("records")
        if not self.rows: raise ValueError(f"no patches in split={split}")
        self.config, self.augment = config, augment
        self.ignore = int(config["data"]["ignore_index"])

    def __len__(self): return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row=self.rows[index]; mask=decode_gt_patch(Path(row["gt_path"]),int(row["x_10x"]),int(row["y_10x"]),int(row["width_10x"]),int(row["height_10x"]),self.config["data"]["class_map"],self.ignore)
        image=read_he_patch(Path(row["wsi_path"]),row["x_level0"],row["y_level0"],row["width_level0"],row["height_level0"],row["width_10x"])
        if image.shape[:2] != mask.shape: raise RuntimeError(f"HE/GT shape mismatch for {row['patch_id']}: {image.shape} vs {mask.shape}")
        if self.augment:
            # all geometric transforms are jointly applied; no interpolation ever touches labels
            k=int(torch.randint(0,4,()).item()); image=np.rot90(image,k).copy(); mask=np.rot90(mask,k).copy()
            if torch.rand(()) < .5: image=np.fliplr(image).copy(); mask=np.fliplr(mask).copy()
            if torch.rand(()) < .5: image=np.flipud(image).copy(); mask=np.flipud(mask).copy()
            gain=float(torch.empty(()).uniform_(.9,1.1)); image=np.clip(image.astype(np.float32)*gain,0,255).astype(np.uint8)
        tensor=torch.from_numpy(image.transpose(2,0,1)).float().div_(255)
        tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        return {"image":tensor,"mask":torch.from_numpy(mask.astype(np.int64)),"valid_mask":torch.from_numpy(mask != self.ignore),"wsi_id":row["wsi_id"],"patch_id":row["patch_id"],"coords_10x":torch.tensor([row["x_10x"],row["y_10x"],row["width_10x"],row["height_10x"]])}


class MultiScaleSegmentationDataset(SegmentationDataset):
    """Same-center 10x/5x/2.5x HE crops; the 10x GT remains supervision."""
    def __init__(self, index_path, config, split, scales=("10x", "5x", "2p5x"), augment=False):
        super().__init__(index_path, config, split, augment=False); self.scales=tuple(scales); self.augment=augment
        if not set(self.scales).issubset({"10x","5x","2p5x"}): raise ValueError(f"unsupported scales: {self.scales}")
    def __getitem__(self,index):
        row=self.rows[index]; mask=decode_gt_patch(Path(row['gt_path']),int(row['x_10x']),int(row['y_10x']),int(row['width_10x']),int(row['height_10x']),self.config['data']['class_map'],self.ignore)
        images=[]
        for scale in self.scales:
            prefix='' if scale=='10x' else f'_{scale}'
            images.append(read_he_patch(Path(row['wsi_path']),int(row[f'x{prefix}_level0']),int(row[f'y{prefix}_level0']),int(row[f'width{prefix}_level0']),int(row[f'height{prefix}_level0']),int(row['width_10x'])))
        if self.augment:
            k=int(torch.randint(0,4,()).item()); images=[np.rot90(x,k).copy() for x in images]; mask=np.rot90(mask,k).copy()
            if torch.rand(())<.5: images=[np.fliplr(x).copy() for x in images]; mask=np.fliplr(mask).copy()
            if torch.rand(())<.5: images=[np.flipud(x).copy() for x in images]; mask=np.flipud(mask).copy()
        x=np.concatenate(images,axis=2); tensor=torch.from_numpy(x.transpose(2,0,1)).float().div_(255)
        mean=torch.tensor([.485,.456,.406]*len(self.scales))[:,None,None]; std=torch.tensor([.229,.224,.225]*len(self.scales))[:,None,None]
        return {'image':(tensor-mean)/std,'mask':torch.from_numpy(mask.astype(np.int64)),'wsi_id':row['wsi_id'],'patch_id':row['patch_id']}


class BalancedPatchSampler(Sampler[int]):
    """Deterministic, rank-disjoint quota sampler for DDP training."""
    def __init__(self, dataset: SegmentationDataset, ratios: dict[str, float], seed: int, rank: int = 0, world_size: int = 1, epoch_size: int | None = None):
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid DDP rank/world_size: {rank}/{world_size}")
        self.groups={name:np.asarray([i for i,r in enumerate(dataset.rows) if r["sampling_group"]==name]) for name in ratios}
        absent=[name for name, ids in self.groups.items() if not len(ids)]
        if absent: raise ValueError(f"configured sampling groups have no patches: {absent}; update config explicitly")
        if not np.isclose(sum(ratios.values()),1): raise ValueError("sampling group ratios must sum to 1")
        if epoch_size is not None and epoch_size < 1:
            raise ValueError("epoch_size must be positive when set")
        self.ratios,self.seed,self.rank,self.world_size=ratios,seed,rank,world_size
        requested=epoch_size if epoch_size is not None else len(dataset)
        self.global_length=(requested//world_size)*world_size
        if not self.global_length:
            raise ValueError(f"epoch size has fewer patches than DDP workers: {requested} < {world_size}")
        self.length=self.global_length//world_size
        self.epoch=0

    def set_epoch(self, epoch: int) -> None:
        self.epoch=epoch

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); chunks=[]; remaining=self.global_length
        for position,(name,ratio) in enumerate(self.ratios.items()):
            count=remaining if position==len(self.ratios)-1 else round(self.global_length*ratio)
            chunks.append(rng.choice(self.groups[name],size=count,replace=True)); remaining-=count
        ids=np.concatenate(chunks); rng.shuffle(ids)
        return iter(ids[self.rank:self.global_length:self.world_size].tolist())
    def __len__(self): return self.length
