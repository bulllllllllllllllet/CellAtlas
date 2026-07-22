from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import torch


CELL_CLASS_COUNT = 7


def encode_xcell_features(cells: np.ndarray, reg_features: np.ndarray) -> np.ndarray:
    """Encode non-ordinal cell metadata and concatenate XCellFormer features.

    Raw metadata is ``[x, y, log1p(area), nuclei_class]``.  The nuclei class is
    categorical, so feeding its integer ID as a continuous scalar would impose a
    false ordering.  Keep the cache schema stable and one-hot encode at load time.
    """
    cells=np.asarray(cells,dtype=np.float32); reg_features=np.asarray(reg_features,dtype=np.float32)
    if cells.ndim!=2 or cells.shape[1]!=4: raise ValueError(f"expected cell metadata [N,4], got {cells.shape}")
    if reg_features.ndim!=2 or reg_features.shape[1]!=64: raise ValueError(f"expected XCell features [N,64], got {reg_features.shape}")
    if len(cells)!=len(reg_features): raise ValueError("cell metadata/XCell feature cardinality mismatch")
    class_ids=cells[:,3]
    if not np.all(np.isfinite(cells)) or not np.all(np.isfinite(reg_features)): raise ValueError("cell features contain non-finite values")
    if not np.allclose(class_ids,np.rint(class_ids)) or np.any(class_ids<0) or np.any(class_ids>=CELL_CLASS_COUNT):
        raise ValueError(f"nuclei class IDs must be integers in [0,{CELL_CLASS_COUNT-1}]")
    one_hot=np.eye(CELL_CLASS_COUNT,dtype=np.float32)[class_ids.astype(np.int64)]
    return np.concatenate((cells[:,:3],one_hot,reg_features),axis=1)


def load_cell_cache(shards: list[str]) -> dict[str, tuple[np.ndarray, int]]:
    table=pd.concat([pd.read_parquet(path,columns=["patch_id","cells","cell_count"]) for path in shards],ignore_index=True)
    if table.patch_id.duplicated().any(): raise ValueError("duplicate patch_id in cell cache")
    return {row.patch_id:(np.asarray(row.cells,dtype=np.float32),int(row.cell_count)) for row in table.itertuples(index=False)}


def load_xcell_feature_cache(shards: list[str]) -> dict[str, tuple[np.ndarray, int]]:
    """Load `[x,y,area,class] + XCellFormer reg-64` without losing the true cell density."""
    table=pd.concat([pd.read_parquet(path,columns=["patch_id","cells","reg_features","total_cell_count"]) for path in shards],ignore_index=True)
    if table.patch_id.duplicated().any(): raise ValueError("duplicate patch_id in XCell feature cache")
    output={}
    for row in table.itertuples(index=False):
        cells=np.stack(row.cells).astype(np.float32,copy=False) if len(row.cells) else np.empty((0,4),np.float32)
        reg=np.stack(row.reg_features).astype(np.float32,copy=False) if len(row.reg_features) else np.empty((0,64),np.float32)
        try: features=encode_xcell_features(cells,reg)
        except ValueError as exc: raise ValueError(f"{row.patch_id}: {exc}") from exc
        output[row.patch_id]=(features,int(row.total_cell_count))
    return output


def collate_cells(items: list[tuple[str,np.ndarray,int]], max_cells: int) -> dict[str,torch.Tensor]:
    if max_cells < 1: raise ValueError("max_cells must be positive")
    arrays=[]
    for patch_id,cells,total in items:
        cells=np.asarray(cells,dtype=np.float32)
        if cells.ndim!=2: raise ValueError(f"{patch_id}: cells must have shape [N,D], got {cells.shape}")
        if len(cells)>max_cells:
            seed=int(hashlib.sha256(patch_id.encode()).hexdigest()[:16],16); ids=np.random.default_rng(seed).choice(len(cells),max_cells,replace=False); cells=cells[np.sort(ids)]
        arrays.append((cells,total))
    width=max((len(x[0]) for x in arrays),default=0); feature_dim=arrays[0][0].shape[1] if arrays else 0
    if any(cells.shape[1]!=feature_dim for cells,_ in arrays): raise ValueError("all patches in a batch must have the same cell feature dimension")
    out=torch.zeros(len(arrays),width,feature_dim); valid=torch.zeros(len(arrays),width,dtype=torch.bool); total=torch.tensor([x[1] for x in arrays],dtype=torch.float32)
    for i,(cells,_) in enumerate(arrays): out[i,:len(cells)]=torch.from_numpy(cells); valid[i,:len(cells)]=True
    return {"cells":out,"cell_valid":valid,"total_cell_count":total}
