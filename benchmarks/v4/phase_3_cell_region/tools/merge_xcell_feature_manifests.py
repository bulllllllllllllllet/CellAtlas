"""Merge completed XCell feature shards and reject gaps/overlaps before training."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Merge Phase 3 XCell feature shard indexes without rewriting data.")
    p.add_argument("--feature-root",action="append",required=True)
    p.add_argument("--split",choices=("train","val","test"),required=True)
    p.add_argument("--expected-rows",type=int,required=True)
    p.add_argument("--output-root",required=True); p.add_argument("--timestamp",default=None)
    return p.parse_args()


def main() -> None:
    ns=parse(); stamp=ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(ns.output_root)/f"xcell_feature_manifest_{ns.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    frames=[]
    for text in ns.feature_root:
        root=Path(text); metadata=root/"metadata.json"; index=root/"feature_index.parquet"
        if metadata.exists() and json.loads(metadata.read_text())["split"]!=ns.split: raise ValueError(f"split mismatch in {root}")
        if index.exists(): frames.append(pd.read_parquet(index)); continue
        recovered=[]
        for shard in sorted(root.glob("features_*.parquet")):
            match=re.fullmatch(r"features_(\d+)_(\d+)\.parquet",shard.name)
            if match is None: raise ValueError(f"unparseable shard name: {shard}")
            start,last=map(int,match.groups()); recovered.append({"shard_path":str(shard),"start":start,"end":last+1,"rows":last-start+1})
        if not recovered: raise ValueError(f"no feature index or shards in {root}")
        frames.append(pd.DataFrame(recovered))
    manifest=pd.concat(frames,ignore_index=True).sort_values("start").reset_index(drop=True)
    contiguous=(len(manifest)>0 and int(manifest.start.iloc[0])==0 and int(manifest.end.iloc[-1])==ns.expected_rows and not manifest.start.duplicated().any() and (len(manifest)==1 or (manifest.start.iloc[1:].to_numpy()==manifest.end.iloc[:-1].to_numpy()).all()))
    if not contiguous or int(manifest.rows.sum())!=ns.expected_rows: raise ValueError(f"feature coverage is not contiguous 0:{ns.expected_rows}")
    out.mkdir(parents=True); manifest.to_parquet(out/"feature_manifest.parquet",index=False)
    (out/"metadata.json").write_text(json.dumps({"split":ns.split,"expected_rows":ns.expected_rows,"shard_count":len(manifest),"source_roots":ns.feature_root},indent=2))
    print({"event":"complete","output":str(out),"rows":int(manifest.rows.sum()),"shards":len(manifest)},flush=True)


if __name__=="__main__": main()
