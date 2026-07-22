#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir,load_config,save_json
from benchmarks.v4.phase_1_multiscale.src.dataset import SegmentationDataset
from benchmarks.v4.phase_1_multiscale.src.metrics import confusion_matrix,summarize
from benchmarks.v4.phase_1_multiscale.src.model import build_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--patch-index',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--split',default='test');p.add_argument('--timestamp');a=p.parse_args();c=load_config(a.config);out=create_run_dir(c,'eval',a.timestamp)
 d=SegmentationDataset(a.patch_index,c,a.split);m=build_model(c,len(c['data']['class_map']));m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);m.eval();conf=None
 for b in DataLoader(d,batch_size=c['training']['batch_size_per_gpu'],num_workers=c['training']['num_workers']):
  with torch.no_grad(): pred=m(b['image'])['out'].argmax(1).numpy()
  now=confusion_matrix(b['mask'].numpy(),pred,len(c['data']['class_map']),c['data']['ignore_index']);conf=now if conf is None else conf+now
 save_json(out/'metrics.json',summarize(conf,set(c['data']['background_class_ids'])));print(out)
if __name__=='__main__':main()
