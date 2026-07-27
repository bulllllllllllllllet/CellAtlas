"""Derive a pure one-positive/one-negative point-prompt manifest."""
import argparse, json
from pathlib import Path
import pandas as pd
from benchmarks.v4.baseline.common import atomic_json, new_output_directory, sha256_path, timestamp, validate_episode_manifest

p=argparse.ArgumentParser(); p.add_argument('--episode-manifest',type=Path,required=True); p.add_argument('--split',required=True); p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/v4/baseline')); p.add_argument('--timestamp'); a=p.parse_args()
s=timestamp(a.timestamp); out=new_output_directory(a.output_root,'one_point_one_negative_manifest',s); f=pd.read_parquet(a.episode_manifest).copy()
for i,row in f.iterrows():
 pos=json.loads(row.positive_points_10x); neg=json.loads(row.negative_points_10x)
 if not pos or not neg: raise ValueError(f'episode {row.occurrence_id} lacks positive/negative point')
 f.at[i,'positive_points_10x']=json.dumps([pos[0]],separators=(',',':')); f.at[i,'negative_points_10x']=json.dumps([neg[0]],separators=(',',':')); f.at[i,'prompt_size']='point'; f.at[i,'positive_box_10x']=None
audit=validate_episode_manifest(f,a.split); path=out/f'episode_manifest_{s}.parquet'; f.to_parquet(path,index=False); atomic_json(out/f'metadata_{s}.json',{'timestamp':s,'source':str(a.episode_manifest),'source_sha256':sha256_path(a.episode_manifest),'manifest':str(path),'manifest_sha256':sha256_path(path),'audit':audit,'rule':'first_frozen_positive_and_first_frozen_negative_point_no_box'})
print(json.dumps({'manifest':str(path),'audit':audit},indent=2))
