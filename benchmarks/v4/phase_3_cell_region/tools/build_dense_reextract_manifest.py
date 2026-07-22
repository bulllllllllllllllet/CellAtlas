"""Build the exact source-patch list that needs spatial-stratified XCell feature re-extraction."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Select only XCell patches whose true cell count exceeds the encoding cap.")
    p.add_argument("--feature-manifest",required=True); p.add_argument("--patch-index",required=True); p.add_argument("--split",choices=("train","val","test"),required=True)
    p.add_argument("--max-cells",type=int,default=255); p.add_argument("--output-root",required=True); p.add_argument("--timestamp",default=None)
    return p.parse_args()


def main() -> None:
    args=parse(); stamp=args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(args.output_root)/f"dense_reextract_{args.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    index=pd.read_parquet(args.patch_index); index=index[index.split==args.split].reset_index(drop=True)
    manifest=pd.read_parquet(args.feature_manifest).sort_values("start").reset_index(drop=True)
    if len(manifest)==0 or int(manifest.start.iloc[0])!=0 or int(manifest.end.iloc[-1])!=len(index) or (len(manifest)>1 and not (manifest.start.iloc[1:].to_numpy()==manifest.end.iloc[:-1].to_numpy()).all()): raise ValueError("feature manifest is not contiguous for its source split")
    selected=[]
    for shard in manifest.itertuples(index=False):
        counts=pd.read_parquet(shard.shard_path,columns=["total_cell_count"])["total_cell_count"].to_numpy()
        if len(counts)!=int(shard.rows): raise ValueError(f"row mismatch: {shard.shard_path}")
        local=np.flatnonzero(counts>args.max_cells)
        if len(local): selected.extend((local+int(shard.start)).tolist())
    source=np.asarray(selected,dtype=np.int64); result=index.iloc[source].copy(); result.insert(0,"source_index",source)
    out.mkdir(parents=True); result.to_parquet(out/"patch_index.parquet",index=False)
    summary={"split":args.split,"source_rows":len(index),"dense_rows":len(result),"dense_fraction":float(len(result)/len(index)) if len(index) else 0.0,"max_cells":args.max_cells,"source_feature_manifest":args.feature_manifest}
    (out/"metadata.json").write_text(json.dumps(summary,indent=2)); print({"event":"complete","output":str(out),**summary},flush=True)


if __name__=="__main__": main()
