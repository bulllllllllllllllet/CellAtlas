from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Merge validated Phase 3 cell-cache shard indexes without rewriting shards.")
    p.add_argument("--cache-root",action="append",required=True)
    p.add_argument("--split",choices=("train","val"),required=True)
    p.add_argument("--expected-rows",type=int,required=True)
    p.add_argument("--output-root",required=True); p.add_argument("--timestamp",default=None)
    return p.parse_args()


def main() -> None:
    ns=parse(); stamp=ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(ns.output_root)/f"cell_cache_manifest_{ns.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    frames=[]
    for root_text in ns.cache_root:
        root=Path(root_text); metadata_path=root/"metadata.json"; index_path=root/"cache_index.parquet"
        if metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            if metadata["split"]!=ns.split: raise ValueError(f"split mismatch in {root}: {metadata['split']}")
        if index_path.exists(): frames.append(pd.read_parquet(index_path)); continue
        recovered=[]
        for shard in sorted(root.glob("cells_*.parquet")):
            match=re.fullmatch(r"cells_(\d+)_(\d+)\.parquet",shard.name)
            if match is None: raise ValueError(f"unparseable shard filename: {shard}")
            start,last=map(int,match.groups()); recovered.append({"shard_path":str(shard),"start":start,"end":last+1,"rows":last-start+1})
        if not recovered: raise ValueError(f"no cache index or shard files in {root}")
        frames.append(pd.DataFrame(recovered))
    manifest=pd.concat(frames,ignore_index=True).sort_values("start").reset_index(drop=True)
    if manifest.start.duplicated().any(): raise ValueError("duplicate shard starts")
    if int(manifest.start.iloc[0])!=0 or int(manifest.end.iloc[-1])!=ns.expected_rows or not (manifest.start.iloc[1:].to_numpy()==manifest.end.iloc[:-1].to_numpy()).all():
        raise ValueError(f"cache coverage is not contiguous 0:{ns.expected_rows}")
    if int(manifest.rows.sum())!=ns.expected_rows: raise ValueError("row total mismatch")
    out.mkdir(parents=True); manifest.to_parquet(out/"cache_manifest.parquet",index=False)
    (out/"metadata.json").write_text(json.dumps({"split":ns.split,"expected_rows":ns.expected_rows,"shard_count":len(manifest),"source_roots":ns.cache_root},indent=2))
    print({"event":"complete","output":str(out),"rows":int(manifest.rows.sum()),"shards":len(manifest)},flush=True)


if __name__=="__main__": main()
