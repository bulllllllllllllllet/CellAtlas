#!/usr/bin/env python3
"""Convert completed compressed SLIC shards to uint8 mmap arrays without recomputing SLIC."""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))


def parse():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--workers",type=int,default=4); p.add_argument("--limit-shards",type=int); p.add_argument("--resume",action="store_true"); return p.parse_args()


def convert_one(source_npz: Path, output_shards: Path) -> dict:
    with np.load(source_npz) as archive: labels=archive["labels"]
    if labels.ndim != 3 or labels.shape[1:] != (512,512): raise ValueError(f"unexpected labels shape: {source_npz}: {labels.shape}")
    maximum=int(labels.max())
    if maximum > 255: raise ValueError(f"SLIC label exceeds uint8 in {source_npz}: {maximum}")
    output_npy=output_shards/f"{source_npz.stem}.npy"; np.save(output_npy,labels.astype(np.uint8,copy=False),allow_pickle=False)
    source_index=source_npz.with_suffix(".parquet"); rows=pd.read_parquet(source_index); rows["shard_path"]=str(output_npy)
    output_index=output_shards/source_index.name; rows.to_parquet(output_index,index=False)
    return {"shard":source_npz.stem,"count":len(rows),"max_label":maximum,"array_path":str(output_npy),"index_path":str(output_index)}


def main():
    ns=parse()
    if ns.output.exists() and not ns.resume: raise FileExistsError(f"refusing to overwrite output: {ns.output}; pass --resume only for the same interrupted conversion")
    if ns.workers < 1: raise ValueError("workers must be positive")
    source_shards=sorted((ns.source/"shards").glob("slic_*.npz"))
    if ns.limit_shards: source_shards=source_shards[:ns.limit_shards]
    if not source_shards: raise ValueError("no source SLIC npz shards")
    ns.output.mkdir(parents=True,exist_ok=ns.resume); output_shards=ns.output/"shards"; output_shards.mkdir(exist_ok=ns.resume); completed=ns.output/"completed.jsonl"; failures=ns.output/"failures.jsonl"
    done=set()
    if ns.resume:
        if not completed.is_file(): raise FileNotFoundError(f"resume requested without completed.jsonl: {completed}")
        done={json.loads(line)["shard"] for line in completed.read_text(encoding="utf-8").splitlines() if line.strip()}; source_shards=[path for path in source_shards if path.stem not in done]
    records=[]
    mode="a" if ns.resume else "x"
    with completed.open(mode,encoding="utf-8") as ok, failures.open(mode,encoding="utf-8") as bad, ProcessPoolExecutor(max_workers=ns.workers) as pool:
        futures={pool.submit(convert_one,path,output_shards):path for path in source_shards}
        for future in as_completed(futures):
            try: record=future.result(); records.append(record); ok.write(json.dumps(record)+"\n"); ok.flush()
            except Exception as exc:
                failure={"source":str(futures[future]),"error":repr(exc)}; bad.write(json.dumps(failure)+"\n"); bad.flush(); raise
    parts=[pd.read_parquet(path) for path in sorted(output_shards.glob("slic_*.parquet"))]; index=pd.concat(parts,ignore_index=True); index.to_parquet(ns.output/"slic_index.parquet",index=False)
    all_records=[json.loads(line) for line in completed.read_text(encoding="utf-8").splitlines() if line.strip()]; max_label=max(record["max_label"] for record in all_records)
    metadata={"source":str(ns.source),"format":"uint8_npy_mmap","shards":len(parts),"patch_count":len(index),"max_label":max_label}; (ns.output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
