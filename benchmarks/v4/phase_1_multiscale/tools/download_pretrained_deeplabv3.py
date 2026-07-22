#!/usr/bin/env python3
"""Download the official torchvision DeepLabV3-ResNet50 COCO checkpoint to v4 NFS storage."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True);parser.add_argument("--timestamp");args=parser.parse_args()
    cfg=load_config(args.config); out=create_run_dir(cfg,"pretrained_weights",args.timestamp)
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    path=torch.hub.load_state_dict_from_url(weights.url, model_dir=str(out), progress=True, check_hash=True)
    save_json(out/"metadata.json",{"architecture":"deeplabv3_resnet50","source":"torchvision","weights":"COCO_WITH_VOC_LABELS_V1","url":weights.url,"checkpoint":str(path)})
    print(path)


if __name__ == "__main__": main()
