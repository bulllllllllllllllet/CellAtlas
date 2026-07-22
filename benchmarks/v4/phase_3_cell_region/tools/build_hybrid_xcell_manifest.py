"""Create one validated per-patch route: spatial-stratified dense features override legacy sparse features."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Build a Phase 3 hybrid XCell feature routing manifest without copying feature shards.")
    p.add_argument("--base-manifest",required=True); p.add_argument("--dense-patch-index",required=True); p.add_argument("--dense-root",action="append",required=True)
    p.add_argument("--split",choices=("train","val","test"),required=True); p.add_argument("--expected-rows",type=int,required=True); p.add_argument("--output-root",required=True); p.add_argument("--timestamp",default=None)
    return p.parse_args()


def validate_contiguous(manifest: pd.DataFrame, expected: int) -> None:
    manifest=manifest.sort_values("start").reset_index(drop=True)
    if len(manifest)==0 or int(manifest.start.iloc[0])!=0 or int(manifest.end.iloc[-1])!=expected or (len(manifest)>1 and not (manifest.start.iloc[1:].to_numpy()==manifest.end.iloc[:-1].to_numpy()).all()) or int(manifest.rows.sum())!=expected:
        raise ValueError(f"base manifest is not contiguous 0:{expected}")


def main() -> None:
    args=parse(); stamp=args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"); out=Path(args.output_root)/f"xcell_hybrid_manifest_{args.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    base=pd.read_parquet(args.base_manifest).sort_values("start").reset_index(drop=True); validate_contiguous(base,args.expected_rows)
    dense_expected=pd.read_parquet(args.dense_patch_index,columns=["source_index","patch_id"])
    if dense_expected.source_index.duplicated().any() or (dense_expected.source_index<0).any() or (dense_expected.source_index>=args.expected_rows).any(): raise ValueError("invalid dense source_index list")
    dense_records=[]
    for root_text in args.dense_root:
        root=Path(root_text); meta=json.loads((root/"metadata.json").read_text())
        if meta.get("split")!=args.split or meta.get("selection_policy")!="spatial_stratified": raise ValueError(f"not a spatial dense root for {args.split}: {root}")
        for shard in pd.read_parquet(root/"feature_index.parquet").itertuples(index=False):
            values=pd.read_parquet(shard.shard_path,columns=["source_index","patch_id"])
            if len(values)!=int(shard.rows) or values.source_index.isna().any(): raise ValueError(f"invalid dense shard: {shard.shard_path}")
            values=values.copy(); values["feature_shard_path"]=str(shard.shard_path); values["row_offset"]=np.arange(len(values),dtype=np.int32); dense_records.append(values)
    dense=pd.concat(dense_records,ignore_index=True) if dense_records else pd.DataFrame(columns=["source_index","patch_id","feature_shard_path","row_offset"])
    if dense.source_index.duplicated().any() or len(dense)!=len(dense_expected): raise ValueError("dense shard count or source_index uniqueness mismatch")
    expected=dense_expected.sort_values("source_index").reset_index(drop=True); actual=dense[["source_index","patch_id"]].sort_values("source_index").reset_index(drop=True)
    if not (np.array_equal(expected.source_index.to_numpy(dtype=np.int64),actual.source_index.to_numpy(dtype=np.int64)) and np.array_equal(expected.patch_id.to_numpy(dtype=str),actual.patch_id.to_numpy(dtype=str))):
        raise ValueError("dense patch ids do not exactly match the planned re-extract list")
    source=np.arange(args.expected_rows,dtype=np.int64); positions=np.searchsorted(base.end.to_numpy(),source,side="right"); route=pd.DataFrame({"source_index":source,"feature_shard_path":base.shard_path.to_numpy()[positions],"row_offset":source-base.start.to_numpy()[positions],"feature_origin":"legacy_sparse"})
    dense_route=dense.set_index("source_index")
    route.loc[dense_route.index,"feature_shard_path"]=dense_route.feature_shard_path.to_numpy(); route.loc[dense_route.index,"row_offset"]=dense_route.row_offset.to_numpy(); route.loc[dense_route.index,"feature_origin"]="spatial_stratified_dense"
    if route.feature_shard_path.isna().any() or route.row_offset.isna().any() or len(route)!=args.expected_rows: raise ValueError("incomplete hybrid route")
    out.mkdir(parents=True); route.to_parquet(out/"feature_routing.parquet",index=False)
    summary={"split":args.split,"rows":args.expected_rows,"dense_spatial_rows":len(dense),"legacy_sparse_rows":args.expected_rows-len(dense),"base_manifest":args.base_manifest,"dense_patch_index":args.dense_patch_index,"dense_roots":args.dense_root}
    (out/"metadata.json").write_text(json.dumps(summary,indent=2)); print({"event":"complete","output":str(out),**summary},flush=True)


if __name__=="__main__": main()
