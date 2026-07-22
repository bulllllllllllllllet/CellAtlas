#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.dataset import BalancedPatchSampler
from benchmarks.v4.phase_2_region_encoder.src.dataset import RegionDataset
from benchmarks.v4.phase_2_region_encoder.src.losses import RegionizationLoss
from benchmarks.v4.phase_2_region_encoder.src.metrics import hard_region_metrics
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder


def parse():
    p=argparse.ArgumentParser(description="DDP training for Phase 2 learned regionization")
    p.add_argument("--config",type=Path,required=True); p.add_argument("--patch-index",type=Path,required=True); p.add_argument("--slic-index",type=Path,required=True)
    p.add_argument("--timestamp"); p.add_argument("--max-epochs",type=int); p.add_argument("--samples-per-epoch",type=int); p.add_argument("--num-workers",type=int); p.add_argument("--limit-train-patches",type=int); p.add_argument("--limit-val-patches",type=int); p.add_argument("--log-every-steps",type=int,default=100); p.add_argument("--resume",type=Path); return p.parse_args()


def seed(value: int):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def save_json(path: Path, value: object):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def stratified_ids(rows: list[dict], limit: int, seed_value: int) -> list[int]:
    if not 1 <= limit <= len(rows): raise ValueError(f"patch limit must be in 1..{len(rows)}")
    groups={name:[] for name in sorted({row["sampling_group"] for row in rows})}
    for index,row in enumerate(rows): groups[row["sampling_group"]].append(index)
    if limit < len(groups): raise ValueError("validation limit is smaller than the number of sampling groups")
    rng=np.random.default_rng(seed_value); selected={int(rng.choice(indices)) for indices in groups.values()}; remaining=np.asarray(sorted(set(range(len(rows)))-selected))
    if limit > len(selected): selected.update(map(int,rng.choice(remaining,size=limit-len(selected),replace=False)))
    result=np.asarray(sorted(selected)); rng.shuffle(result); return result.tolist()


def select_rows(dataset: RegionDataset, indices: list[int]) -> None:
    dataset.rows=[dataset.rows[index] for index in indices]; dataset.refs=[dataset.refs[index] for index in indices]


def all_ranks_true(value: bool, device: torch.device, world: int) -> bool:
    flag=torch.tensor(1 if value else 0,device=device,dtype=torch.int32)
    if world>1: dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main():
    ns=parse(); cfg=load_config(ns.config); rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    if world>1: dist.init_process_group("nccl"); torch.cuda.set_device(rank)
    device=torch.device("cuda",rank); seed(int(cfg["project"]["seed"])+rank)
    epochs=ns.max_epochs or int(cfg["training"]["epochs"]); workers=int(ns.num_workers if ns.num_workers is not None else cfg["training"]["num_workers"])
    train=RegionDataset(ns.patch_index,ns.slic_index,cfg,"train",augment=True); val=RegionDataset(ns.patch_index,ns.slic_index,cfg,"val",augment=False)
    if ns.limit_train_patches: select_rows(train,stratified_ids(train.rows,ns.limit_train_patches,int(cfg["project"]["seed"])))
    val_limit=ns.limit_val_patches or cfg["training"].get("validation_samples_per_epoch")
    if val_limit: select_rows(val,stratified_ids(val.rows,int(val_limit),int(cfg["project"]["seed"])+1))
    if world>1: val=Subset(val,list(range(rank,len(val),world)))
    budget=ns.samples_per_epoch or int(cfg["training"]["samples_per_epoch"])
    sampler=BalancedPatchSampler(train,{"class_interior":.3,"class_boundary":.3,"rare_class":.3,"background_or_hard_negative":.1},int(cfg["project"]["seed"]),rank,world,budget)
    kwargs={"num_workers":workers,"pin_memory":True,"persistent_workers":bool(workers)}
    if workers: kwargs.update({"multiprocessing_context":"spawn","prefetch_factor":1})
    loader=DataLoader(train,batch_size=cfg["training"]["batch_size_per_gpu"],sampler=sampler,**kwargs); vloader=DataLoader(val,batch_size=cfg["training"]["batch_size_per_gpu"],shuffle=False,**kwargs)
    classes=len(cfg["data"]["class_map"]); model=DeepRegionEncoder(cfg,classes,int(cfg["model"]["num_regions"]),int(cfg["model"]["embedding_dim"])).to(device)
    loss_fn=RegionizationLoss(classes,int(cfg["data"]["ignore_index"]),set(cfg["data"]["background_class_ids"]),cfg["training"]["loss_weights"])
    pretrained_params=list(model.backbone.parameters())+list(model.semantic.parameters()); region_params=list(model.embedding.parameters())+list(model.assignment.parameters())
    opt=torch.optim.AdamW([{"params":pretrained_params,"lr":cfg["training"]["pretrained_lr"]},{"params":region_params,"lr":cfg["training"]["region_head_lr"]}],weight_decay=cfg["training"]["weight_decay"])
    amp_dtype={"float16":torch.float16,"bfloat16":torch.bfloat16}[cfg["training"]["amp_dtype"]]; scaler=torch.amp.GradScaler("cuda",enabled=bool(cfg["training"]["amp"]) and amp_dtype is torch.float16)
    start_epoch=0; history=[]
    if ns.resume:
        checkpoint=torch.load(ns.resume,map_location=device,weights_only=False); model.load_state_dict(checkpoint["model"]); opt.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"]); start_epoch=int(checkpoint["epoch"])+1; history=list(checkpoint.get("history",[]))
    if world>1: model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank])
    stamp=ns.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"); out=Path(cfg["output"]["root"])/f"phase_2_region_encoder_train_{stamp}"
    if not all_ranks_true(not out.exists(),device,world): raise FileExistsError(f"refusing to overwrite existing run: {out}")
    if rank==0:
        out.mkdir(parents=True); save_json(out/"config_snapshot.json",cfg); save_json(out/"run_metadata.json",{"patch_index":str(ns.patch_index),"slic_index":str(ns.slic_index),"world_size":world,"batch_size_per_gpu":cfg["training"]["batch_size_per_gpu"],"num_workers_per_rank":workers,"samples_per_epoch_global":budget,"validation_samples_global":val_limit,"epochs":epochs,"resume":str(ns.resume) if ns.resume else None})
        print({"event":"train_start","output":str(out),"world_size":world,"train_steps":len(loader),"val_steps":len(vloader),"start_epoch":start_epoch,"epochs":epochs,"learning_rates":[group["lr"] for group in opt.param_groups]},flush=True)
    if world>1: dist.barrier()
    for epoch in range(start_epoch,epochs):
        sampler.set_epoch(epoch); model.train(); running=0.; running_parts={}
        hold=int(cfg["training"]["slic_hold_epochs"]); decay=max(int(cfg["training"]["slic_decay_epochs"]),1); slic_weight=1.0 if epoch<hold else max(0.,1.-(epoch-hold)/decay)
        for step,batch in enumerate(loader,1):
            x=batch["image"].to(device); y=batch["mask"].to(device); s=batch["slic"].to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=amp_dtype,enabled=bool(cfg["training"]["amp"])): loss,parts=loss_fn(model(x,return_full_assignment=False,return_tokens=False),y,s,slic_weight)
            if not all_ranks_true(bool(torch.isfinite(loss).item()),device,world): raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}; rank={rank}; parts={parts}")
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["max_grad_norm"]),error_if_nonfinite=False)
            if not all_ranks_true(bool(torch.isfinite(grad_norm).item()),device,world): raise FloatingPointError(f"non-finite gradient at epoch={epoch} step={step}; rank={rank}; grad_norm={float(grad_norm)}; parts={parts}")
            scaler.step(opt); scaler.update(); running+=parts["loss"]
            for name,value in parts.items(): running_parts[name]=running_parts.get(name,0.)+value
            if rank==0 and (step%ns.log_every_steps==0 or step==len(loader)): print({"event":"train_progress","epoch":epoch,"step":step,"steps":len(loader),"mean_loss":running/step,"mean_parts":{name:value/step for name,value in running_parts.items()},"slic_weight":slic_weight},flush=True)
        model.eval(); totals=torch.zeros(4,device=device)
        with torch.no_grad():
            for batch in vloader:
                outp=model(batch["image"].to(device),return_full_assignment=False,return_tokens=False); metrics=[]
                hard=F.interpolate(outp["assignment_low"].argmax(1,keepdim=True).float(),size=batch["mask"].shape[-2:],mode="nearest")[:,0].to(torch.uint8).cpu().numpy()
                for regions,y in zip(hard,batch["mask"].numpy(),strict=True): metrics.append(hard_region_metrics(regions,y,int(cfg["data"]["ignore_index"]),classes))
                for m in metrics: totals += torch.tensor([m["region_purity"],m["boundary_f1"],m["oracle_region_dice"],m["active_regions"]],device=device)
        if world>1: dist.all_reduce(totals)
        if rank==0:
            denom=max(len(val)*world,1); row={"epoch":epoch,"train_loss":running/max(len(loader),1),"region_purity":float(totals[0]/denom),"boundary_f1":float(totals[1]/denom),"oracle_region_dice":float(totals[2]/denom),"active_regions":float(totals[3]/denom)}; history.append(row)
            state={"epoch":epoch,"model":model.module.state_dict() if world>1 else model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"history":history,"class_map":cfg["data"]["class_map"]}; checkpoint_path=out/f"checkpoint_epoch_{epoch:03d}.pth"; torch.save(state,checkpoint_path); save_json(out/"last_checkpoint.json",{"epoch":epoch,"path":str(checkpoint_path)})
            if row["oracle_region_dice"] >= max((x["oracle_region_dice"] for x in history),default=-1): save_json(out/"best_checkpoint.json",{"epoch":epoch,"path":str(checkpoint_path),"oracle_region_dice":row["oracle_region_dice"]})
            save_json(out/"metrics.json",history); print(row,flush=True)
    if world>1: dist.destroy_process_group()


if __name__ == "__main__": main()
