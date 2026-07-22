from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import read_pairs


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Build sharded Phase 3 cell features from GDPH nuclei instance/class maps.")
    p.add_argument("--config",required=True); p.add_argument("--patch-index",required=True); p.add_argument("--output-root",required=True)
    p.add_argument("--split",choices=("train","val","test"),required=True); p.add_argument("--limit",type=int); p.add_argument("--start",type=int,default=0); p.add_argument("--end",type=int); p.add_argument("--shard-size",type=int,default=250)
    p.add_argument("--timestamp",default=None); return p.parse_args()


def extract_cell_features(inst: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Compute all cell centroids, areas, and majority classes in one pixel pass."""
    if inst.shape != cls.shape or inst.ndim != 2:
        raise ValueError(f"instance/class shape mismatch: {inst.shape} vs {cls.shape}")
    positive=inst>0
    if not positive.any(): return np.empty((0,4),dtype=np.float32)
    yy,xx=np.nonzero(positive); ids,inverse=np.unique(inst[positive],return_inverse=True); count=np.bincount(inverse)
    sum_x=np.bincount(inverse,weights=xx); sum_y=np.bincount(inverse,weights=yy)
    class_counts=np.bincount(inverse*7+cls[positive].astype(np.int64),minlength=len(ids)*7).reshape(len(ids),7)
    class_counts[:,0]=0; cell_class=class_counts.argmax(1); cell_class[class_counts.sum(1)==0]=0
    return np.stack((sum_x/count/inst.shape[1],sum_y/count/inst.shape[0],np.log1p(count),cell_class),axis=1).astype(np.float32)


def extract(row: dict, pair) -> dict:
    x,y,w,h=(int(row[k]) for k in ("x_level0","y_level0","width_level0","height_level0"))
    instances=pyvips.Image.new_from_file(str(pair.nuclei_instance_path),access="random").crop(x,y,w,h)
    classes=pyvips.Image.new_from_file(str(pair.nuclei_class_path),access="random").crop(x,y,w,h)
    inst=np.ndarray(buffer=instances.write_to_memory(),dtype=np.int32,shape=(h,w)); cls=np.ndarray(buffer=classes.write_to_memory(),dtype=np.uint8,shape=(h,w))
    cells=extract_cell_features(inst,cls)
    return {"patch_id":row["patch_id"],"wsi_id":row["wsi_id"],"cells":cells.tolist(),"cell_count":len(cells)}


def main() -> None:
    ns=parse(); cfg=load_config(ns.config); stamp=ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(ns.output_root)/f"cell_patch_cache_{ns.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    out.mkdir(parents=True); rows=pd.read_parquet(ns.patch_index).query("split == @ns.split").to_dict("records")
    if ns.start < 0: raise ValueError("--start must be non-negative")
    end=ns.end if ns.end is not None else len(rows)
    if end <= ns.start or end > len(rows): raise ValueError(f"invalid range {ns.start}:{end} for {len(rows)} rows")
    rows=rows[ns.start:end]
    if ns.limit: rows=rows[:ns.limit]
    pairs={x.wsi_id:x for x in read_pairs(cfg)}; missing=sorted({r["wsi_id"] for r in rows}-set(pairs))
    if missing: raise ValueError(f"patch index WSI missing nuclei pair: {missing[:5]} (total {len(missing)})")
    completed=(out/"completed.jsonl").open("a"); refs=[]
    for start in range(0,len(rows),ns.shard_size):
        part=rows[start:start+ns.shard_size]; payload=[extract(row,pairs[row["wsi_id"]]) for row in part]
        absolute_start=ns.start+start; absolute_end=absolute_start+len(part)
        shard=out/f"cells_{absolute_start:07d}_{absolute_end-1:07d}.parquet"; pd.DataFrame(payload).to_parquet(shard,index=False)
        refs.append({"shard_path":str(shard),"start":absolute_start,"end":absolute_end,"rows":len(part)})
        completed.write(json.dumps({"start":absolute_start,"end":absolute_end,"shard_path":str(shard)})+"\n"); completed.flush()
        print({"event":"shard_complete","done":start+len(part),"total":len(rows),"shard":str(shard)},flush=True)
    completed.close(); pd.DataFrame(refs).to_parquet(out/"cache_index.parquet",index=False)
    counts=[x["cell_count"] for shard in refs for x in pd.read_parquet(shard["shard_path"],columns=["cell_count"]).to_dict("records")]
    (out/"metadata.json").write_text(json.dumps({"split":ns.split,"patch_count":len(rows),"feature_schema":"x_norm,y_norm,log1p_area,nuclei_class_id","coordinate_source":"level0 crop mapped to patch-normalized coordinates","cell_count":{"min":min(counts),"max":max(counts),"mean":float(np.mean(counts))}},indent=2))
    print({"event":"complete","output":str(out),"patch_count":len(rows)},flush=True)


if __name__=="__main__": main()
