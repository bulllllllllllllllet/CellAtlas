#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json, seed_everything
from benchmarks.v4.phase_1_multiscale.src.dataset import BalancedPatchSampler, MultiScaleSegmentationDataset, SegmentationDataset
from benchmarks.v4.phase_1_multiscale.src.losses import SegmentationLoss
from benchmarks.v4.phase_1_multiscale.src.metrics import confusion_matrix, summarize
from benchmarks.v4.phase_1_multiscale.src.model import build_model


def parse():
    p=argparse.ArgumentParser(description="10x supervised baseline; use torchrun for DDP.")
    p.add_argument("--config",type=Path,required=True);p.add_argument("--patch-index",type=Path,required=True);p.add_argument("--scales",default="10x",help="comma-separated: 10x,5x,2p5x");p.add_argument("--timestamp");p.add_argument("--max-epochs",type=int);p.add_argument("--overfit-num-patches",type=int);p.add_argument("--limit-train-patches",type=int);p.add_argument("--limit-val-patches",type=int);p.add_argument("--samples-per-epoch",type=int);p.add_argument("--num-workers",type=int);p.add_argument("--log-every-steps",type=int,default=100);p.add_argument("--resume",type=Path);return p.parse_args()


def stratified_limit(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if not 1 <= limit <= len(rows): raise ValueError(f"patch limit must be in 1..{len(rows)}")
    groups={name:[] for name in sorted({row["sampling_group"] for row in rows})}
    for index,row in enumerate(rows): groups[row["sampling_group"]].append(index)
    if limit < len(groups): raise ValueError(f"patch limit {limit} is smaller than {len(groups)} sampling groups")
    rng=np.random.default_rng(seed); selected={int(rng.choice(indices)) for indices in groups.values()}
    remaining=np.asarray(sorted(set(range(len(rows)))-selected)); need=limit-len(selected)
    if need: selected.update(map(int,rng.choice(remaining,size=need,replace=False)))
    ids=np.asarray(sorted(selected)); rng.shuffle(ids)
    return [rows[int(index)] for index in ids]


def main():
    ns=parse();cfg=load_config(ns.config); rank=int(__import__("os").environ.get("RANK",0)); world=int(__import__("os").environ.get("WORLD_SIZE",1))
    if world>1: dist.init_process_group("nccl");torch.cuda.set_device(rank)
    device=torch.device("cuda",rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(cfg["project"]["seed"])+rank); classes=cfg["data"]["class_map"]; ids=[int(c["id"]) for c in classes]
    if ids != list(range(len(ids))): raise ValueError("class ids must be contiguous 0..num_classes-1")
    scales=tuple(ns.scales.split(",")); out=create_run_dir(cfg,"train",ns.timestamp) if rank==0 else None
    dataset=MultiScaleSegmentationDataset if len(scales)>1 else SegmentationDataset
    kwargs={"scales":scales} if dataset is MultiScaleSegmentationDataset else {}
    train=dataset(ns.patch_index,cfg,"train",augment=not ns.overfit_num_patches,**kwargs); val=dataset(ns.patch_index,cfg,"val",**kwargs)
    if ns.overfit_num_patches: train=Subset(train,list(range(min(ns.overfit_num_patches,len(train))))); val=train
    else:
        train_limit=ns.limit_train_patches or cfg["training"].get("limit_train_patches")
        val_limit=ns.limit_val_patches or cfg["training"].get("validation_samples_per_epoch")
        if train_limit: train.rows=stratified_limit(train.rows,int(train_limit),int(cfg["project"]["seed"]))
        if val_limit: val.rows=stratified_limit(val.rows,int(val_limit),int(cfg["project"]["seed"])+1)
        if world>1: val=Subset(val,list(range(rank,len(val),world)))
    epoch_budget=ns.samples_per_epoch or cfg["training"].get("samples_per_epoch")
    sampler=None if isinstance(train,Subset) else BalancedPatchSampler(train,cfg["sampling"]["group_ratios"],int(cfg["project"]["seed"]),rank,world,epoch_budget)
    workers=int(ns.num_workers if ns.num_workers is not None else cfg["training"]["num_workers"])
    if workers < 0: raise ValueError("--num-workers must be non-negative")
    if ns.log_every_steps < 1: raise ValueError("--log-every-steps must be positive")
    loader_kwargs={"num_workers":workers,"pin_memory":True,"persistent_workers":bool(workers)}
    if workers: loader_kwargs.update({"multiprocessing_context":"spawn","prefetch_factor":1})
    loader=DataLoader(train,batch_size=cfg["training"]["batch_size_per_gpu"],sampler=sampler,shuffle=sampler is None,**loader_kwargs)
    vloader=DataLoader(val,batch_size=cfg["training"]["batch_size_per_gpu"],shuffle=False,**loader_kwargs)
    model=build_model(cfg,len(classes),input_channels=3*len(scales)).to(device)
    if world>1:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank])
    loss_fn=SegmentationLoss(len(classes),cfg["data"]["ignore_index"],set(cfg["data"]["background_class_ids"]),cfg["training"]["ce_weight"],cfg["training"]["dice_weight"])
    optim=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],weight_decay=cfg["training"]["weight_decay"]); scaler=torch.amp.GradScaler("cuda",enabled=bool(cfg["training"]["amp"]) and device.type=="cuda")
    start=0
    if ns.resume:
        ck=torch.load(ns.resume,map_location=device,weights_only=False);model.load_state_dict(ck["model"]);optim.load_state_dict(ck["optimizer"]);scaler.load_state_dict(ck["scaler"]);start=ck["epoch"]+1
    epochs=ns.max_epochs or cfg["training"]["epochs"]; history=[]
    if rank==0:
        save_json(out/"config_snapshot.json",cfg)
        save_json(out/"run_metadata.json",{"patch_index":str(ns.patch_index),"scales":scales,"input_channels":3*len(scales),"world_size":world,"num_workers_per_rank":workers,"batch_size_per_rank":cfg["training"]["batch_size_per_gpu"],"train_patches_per_rank":len(sampler) if sampler is not None else len(train),"samples_per_epoch_global":epoch_budget,"val_patches_per_rank":len(val),"epochs":epochs,"amp":bool(cfg["training"]["amp"]),"seed":cfg["project"]["seed"]})
        print({"event":"train_start","output":str(out),"world_size":world,"workers_per_rank":workers,"train_batches_per_rank":len(loader),"val_batches_per_rank":len(vloader),"epochs":epochs},flush=True)
    for epoch in range(start,epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        model.train(); running=0.
        for step,batch in enumerate(loader,1):
            x,y=batch["image"].to(device),batch["mask"].to(device);optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()): logits=model(x)["out"]; loss,parts=loss_fn(logits,y)
            scaler.scale(loss).backward();scaler.step(optim);scaler.update();running+=parts["loss"]
            if rank==0 and (step%ns.log_every_steps==0 or step==len(loader)): print({"event":"train_progress","epoch":epoch,"step":step,"steps":len(loader),"mean_loss":running/step},flush=True)
        model.eval(); conf=torch.zeros((len(classes),len(classes)),dtype=torch.long,device=device)
        with torch.no_grad():
            for batch in vloader:
                pred=model(batch["image"].to(device))["out"].argmax(1).cpu().numpy(); target=batch["mask"].numpy(); conf+=torch.from_numpy(confusion_matrix(target,pred,len(classes),cfg["data"]["ignore_index"])).to(device)
        if world>1:dist.all_reduce(conf)
        if rank==0:
            metric=summarize(conf.cpu().numpy(),set(cfg["data"]["background_class_ids"]));history.append({"epoch":epoch,"train_loss":running/max(len(loader),1),**metric})
            state={"epoch":epoch,"model":model.module.state_dict() if world>1 else model.state_dict(),"optimizer":optim.state_dict(),"scaler":scaler.state_dict(),"class_map":classes,"ignore_index":cfg["data"]["ignore_index"]}
            torch.save(state,out/"last_checkpoint.pth")
            if metric["macro_dice"]>=max((x["macro_dice"] for x in history),default=-1):torch.save(state,out/"best_checkpoint.pth")
            save_json(out/"metrics.json",history)
            print(history[-1],flush=True)
    if world>1:dist.destroy_process_group()

if __name__=="__main__":main()
