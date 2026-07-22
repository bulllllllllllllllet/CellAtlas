#!/usr/bin/env python3
"""Build resumable, deterministic SLIC supervision shards for Phase 2."""
from __future__ import annotations

import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.segmentation import slic

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch


def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--patch-index", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", default="train,val"); p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--n-segments", type=int, default=64); p.add_argument("--compactness", type=float, default=10.)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--limit", type=int); return p.parse_args()


def build_one(row: dict, segments: int, compactness: float) -> tuple[str, str, np.ndarray]:
    image=read_he_patch(Path(row["wsi_path"]), int(row["x_level0"]), int(row["y_level0"]), int(row["width_level0"]), int(row["height_level0"]), int(row["width_10x"]))
    labels=slic(image, n_segments=segments, compactness=compactness, start_label=0, channel_axis=-1, convert2lab=True)
    return row["patch_id"], row["split"], labels.astype(np.uint16)


def main():
    ns=parse()
    if ns.output.exists(): raise FileExistsError(f"refusing to overwrite output: {ns.output}")
    if ns.shard_size < 1 or ns.workers < 1: raise ValueError("shard-size and workers must be positive")
    ns.output.mkdir(parents=True); shard_dir=ns.output/"shards"; shard_dir.mkdir()
    rows=pd.read_parquet(ns.patch_index); rows=rows[rows.split.isin(ns.splits.split(","))].to_dict("records")
    if ns.limit: rows=rows[:ns.limit]
    completed=ns.output/"completed.jsonl"; index_rows=[]
    with completed.open("x", encoding="utf-8") as status:
        for shard_id,start in enumerate(range(0,len(rows),ns.shard_size)):
            chunk=rows[start:start+ns.shard_size]
            with ThreadPoolExecutor(max_workers=ns.workers) as pool:
                packed=list(pool.map(lambda row: build_one(row,ns.n_segments,ns.compactness),chunk))
            labels=np.stack([x[2] for x in packed]); name=f"slic_{shard_id:05d}.npz"; path=shard_dir/name
            np.savez_compressed(path,labels=labels)
            shard_records=[{"patch_id":patch_id,"split":split,"shard_path":str(path),"offset":offset} for offset,(patch_id,split,_) in enumerate(packed)]
            pd.DataFrame(shard_records).to_parquet(shard_dir/f"slic_{shard_id:05d}.parquet",index=False)
            index_rows.extend(shard_records); status.write(json.dumps({"shard_id":shard_id,"count":len(shard_records),"path":str(path)})+"\n"); status.flush()
    pd.DataFrame(index_rows).to_parquet(ns.output/"slic_index.parquet",index=False)
    (ns.output/"metadata.json").write_text(json.dumps({"patch_index":str(ns.patch_index),"splits":ns.splits,"n_segments":ns.n_segments,"compactness":ns.compactness,"patch_count":len(index_rows)},indent=2),encoding="utf-8")


if __name__ == "__main__": main()
