#!/usr/bin/env python3
"""Create a fair same-center 10x/5x/2.5x index without boundary padding."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--patch-index',type=Path,required=True); p.add_argument('--cohort-manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 patches=pd.read_parquet(a.patch_index); slides=pd.read_parquet(a.cohort_manifest)[['wsi_id','level0_width','level0_height']]
 x=patches.merge(slides,on='wsi_id',validate='many_to_one')
 w=x.width_level0.astype(int); h=x.height_level0.astype(int)
 # Same center: 10x=1x field, 5x=2x field, 2.5x=4x field at level 0.
 for scale,mult in [('5x',2),('2p5x',4)]:
  x[f'x_{scale}_level0']=x.x_level0.astype(int)-((mult-1)*w//2)
  x[f'y_{scale}_level0']=x.y_level0.astype(int)-((mult-1)*h//2)
  x[f'width_{scale}_level0']=mult*w; x[f'height_{scale}_level0']=mult*h
 keep=(x.x_2p5x_level0>=0)&(x.y_2p5x_level0>=0)&((x.x_2p5x_level0+x.width_2p5x_level0)<=x.level0_width)&((x.y_2p5x_level0+x.height_2p5x_level0)<=x.level0_height)
 out=x.loc[keep].drop(columns=['level0_width','level0_height']); a.output_dir.mkdir(parents=True)
 out.to_parquet(a.output_dir/'patch_index_10x_5x_2p5x.parquet',index=False)
 meta={'input_patch_count':len(x),'context_valid_patch_count':len(out),'dropped_boundary_patch_count':int((~keep).sum()),'split_counts':out.split.value_counts().to_dict(),'contract':'same center; 10x=1x, 5x=2x, 2.5x=4x level0 field; no synthetic padding'}
 (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta))
if __name__=='__main__': main()
