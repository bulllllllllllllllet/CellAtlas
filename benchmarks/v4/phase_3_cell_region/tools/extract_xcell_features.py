from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import read_pairs
from new_inference_stream import inference as xcell_reference


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Extract HE-only CTransPath + XCellFormer features using supplied GDPH instance masks.")
    p.add_argument("--config",required=True); p.add_argument("--patch-index",required=True); p.add_argument("--output-root",required=True)
    p.add_argument("--ctranspath-checkpoint",default="module/checkpoint/ctranspath.pth"); p.add_argument("--xcell-checkpoint",default="module/checkpoint/he_model_best.pth")
    p.add_argument("--split",choices=("train","val","test"),required=True); p.add_argument("--start",type=int,default=0); p.add_argument("--end",type=int); p.add_argument("--limit",type=int); p.add_argument("--shard-size",type=int,default=25)
    p.add_argument("--max-cells",type=int,default=255); p.add_argument("--cell-batch-size",type=int,default=255)
    p.add_argument(
        "--preprocess-mode",
        choices=("reference","batched"),
        default="batched",
        help="Use batched by default; select reference only for strict historical feature reproduction.",
    )
    p.add_argument("--selection-policy",choices=("legacy_label_prefix","spatial_stratified"),default="legacy_label_prefix"); p.add_argument("--spatial-grid-size",type=int,default=8); p.add_argument("--device",default="cuda:1"); p.add_argument("--timestamp",default=None)
    return p.parse_args()


def spatial_stratified_indices(features: np.ndarray, max_cells: int, grid_size: int) -> np.ndarray:
    """Retain every occupied spatial bin, then allocate remaining slots proportional to bin population."""
    if len(features)<=max_cells: return np.arange(len(features),dtype=np.int64)
    if grid_size<1: raise ValueError("spatial_grid_size must be positive")
    x=np.minimum((features[:,0]*grid_size).astype(np.int64),grid_size-1); y=np.minimum((features[:,1]*grid_size).astype(np.int64),grid_size-1); bins=y*grid_size+x
    unique,counts=np.unique(bins,return_counts=True); groups=[np.flatnonzero(bins==key) for key in unique]
    quota=np.floor(counts*max_cells/len(features)).astype(np.int64); quota=np.maximum(quota,1); quota=np.minimum(quota,counts)
    fractions=counts*max_cells/len(features)-np.floor(counts*max_cells/len(features))
    while quota.sum()<max_cells:
        candidates=np.flatnonzero(quota<counts)
        if not len(candidates): break
        winner=candidates[np.lexsort((unique[candidates],-fractions[candidates]))[0]]; quota[winner]+=1
    while quota.sum()>max_cells:
        candidates=np.flatnonzero(quota>1)
        winner=candidates[np.lexsort((unique[candidates],fractions[candidates]))[0]]; quota[winner]-=1
    selected=[]
    for group,n in zip(groups,quota,strict=True): selected.extend(group[np.linspace(0,len(group)-1,int(n),dtype=np.int64)].tolist())
    return np.asarray(sorted(selected),dtype=np.int64)


def selected_cells_from_maps(inst: np.ndarray, cls: np.ndarray, max_cells: int, selection_policy: str = "legacy_label_prefix", spatial_grid_size: int = 8) -> tuple[np.ndarray,np.ndarray,int]:
    """Select instance cells then return them in y/x order for XCellFormer encoding."""
    valid=inst>0
    if not valid.any(): return np.empty((0,4),np.float32),np.empty((0,),np.int32),0
    yy,xx=np.nonzero(valid); ids,inverse=np.unique(inst[valid],return_inverse=True); count=np.bincount(inverse); sx=np.bincount(inverse,weights=xx); sy=np.bincount(inverse,weights=yy)
    votes=np.bincount(inverse*7+cls[valid].astype(np.int64),minlength=len(ids)*7).reshape(len(ids),7); votes[:,0]=0; label=votes.argmax(1); label[votes.sum(1)==0]=0
    features=np.stack((sx/count/inst.shape[1],sy/count/inst.shape[0],np.log1p(count),label),1).astype(np.float32)
    total=len(ids)
    if selection_policy=="legacy_label_prefix": chosen=np.arange(min(total,max_cells),dtype=np.int64)
    elif selection_policy=="spatial_stratified": chosen=spatial_stratified_indices(features,max_cells,spatial_grid_size)
    else: raise ValueError(f"unknown selection policy: {selection_policy}")
    ids=ids[chosen]; features=features[chosen]
    order=np.lexsort((features[:,0],features[:,1])); return features[order],ids[order].astype(np.int32),total


def load_models(args: argparse.Namespace, device: torch.device):
    return (xcell_reference.load_ctranspath(args.ctranspath_checkpoint,device),
            xcell_reference.load_xcell_model(args.xcell_checkpoint,device))


def open_slide_images(pair) -> tuple[pyvips.Image,pyvips.Image,pyvips.Image]:
    """Keep lazy random-access handles open across adjacent patches from the same WSI."""
    return (pyvips.Image.new_from_file(str(pair.he_path),access="random"),
            pyvips.Image.new_from_file(str(pair.nuclei_instance_path),access="random"),
            pyvips.Image.new_from_file(str(pair.nuclei_class_path),access="random"))


def _vips_array(image: pyvips.Image, dtype: np.dtype, h: int, w: int, bands: int = 1) -> np.ndarray:
    value=image.extract_band(0,n=bands) if bands>1 else image
    shape=(h,w,bands) if bands>1 else (h,w)
    return np.ndarray(buffer=value.write_to_memory(),dtype=dtype,shape=shape)


def extract_raw_reference_compatible(he: np.ndarray, inst: np.ndarray, selected_ids: np.ndarray, ctp, device: torch.device, batch_size: int, preprocess_mode: str) -> tuple[np.ndarray,dict[str,float]]:
    """The established soft-attention recipe, separated only to measure and batch its stages."""
    started=time.perf_counter(); preprocess=xcell_reference.transforms.Compose([
        xcell_reference.transforms.ToTensor(), xcell_reference.transforms.Resize((224,224),interpolation=xcell_reference.Image.BILINEAR),
        xcell_reference.transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),
    ])
    labels=np.asarray(selected_ids,dtype=np.int32)
    if not len(labels): return np.empty((0,768),np.float32),{"preprocess_sec":0.0,"attention_sec":0.0,"transform_sec":0.0,"ctrans_sec":0.0}
    centers=xcell_reference.ndimage.center_of_mass(np.ones_like(inst),labels=inst,index=labels)
    ordered=sorted(zip(labels,centers),key=lambda x:(x[1][0],x[1][1]))
    crops=[]; attended_images=[]; attention_sec=0.0; transform_sec=0.0
    for label,(cy,cx) in ordered:
        cell_started=time.perf_counter()
        crop=xcell_reference.crop_region(he,int(cy),int(cx)); crop_mask=xcell_reference.crop_region_mask(inst,int(cy),int(cx))
        attended=xcell_reference.apply_soft_attention(crop,(crop_mask==label).astype(np.float32))
        attention_sec+=time.perf_counter()-cell_started; cell_started=time.perf_counter(); image=(np.clip(attended,0,1)*255).astype(np.uint8)
        if preprocess_mode=="reference": crops.append(preprocess(xcell_reference.Image.fromarray(image)))
        else: attended_images.append(image)
        transform_sec+=time.perf_counter()-cell_started
    if preprocess_mode=="batched":
        transform_started=time.perf_counter(); batch=torch.from_numpy(np.stack(attended_images)).permute(0,3,1,2).float().div_(255.0)
        crops=F.interpolate(batch,size=(224,224),mode="bilinear",align_corners=False,antialias=True)
        crops=(crops-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)
        transform_sec+=time.perf_counter()-transform_started
    preprocess_sec=time.perf_counter()-started; started=time.perf_counter(); encoded=[]
    with torch.inference_mode():
        for start in range(0,len(crops),batch_size):
            batch=torch.stack(crops[start:start+batch_size]) if isinstance(crops,list) else crops[start:start+batch_size]
            encoded.append(ctp(batch.to(device)).float().cpu())
    return torch.cat(encoded).numpy(),{"preprocess_sec":preprocess_sec,"attention_sec":attention_sec,"transform_sec":transform_sec,"ctrans_sec":time.perf_counter()-started}


FEATURE_SCHEMA=pa.schema([
    pa.field("patch_id",pa.string()), pa.field("wsi_id",pa.string()), pa.field("total_cell_count",pa.int32()),
    pa.field("source_index",pa.int32()),
    pa.field("cells",pa.list_(pa.list_(pa.float32()))),
    pa.field("reg_features",pa.list_(pa.list_(pa.float32()))),
    pa.field("proj_features",pa.list_(pa.list_(pa.float32()))),
])


def extract_one(row: dict, slide_images, ctp, xcell, args, device) -> tuple[dict,dict[str,float]]:
    x,y,w,h=(int(row[k]) for k in ("x_level0","y_level0","width_level0","height_level0"))
    started=time.perf_counter(); he_src,inst_src,cls_src=slide_images
    he=_vips_array(he_src.crop(x,y,w,h),np.uint8,h,w,3); inst=_vips_array(inst_src.crop(x,y,w,h),np.int32,h,w); cls=_vips_array(cls_src.crop(x,y,w,h),np.uint8,h,w)
    read_sec=time.perf_counter()-started
    meta,ids,total=selected_cells_from_maps(inst,cls,args.max_cells,args.selection_policy,args.spatial_grid_size)
    raw,profile=extract_raw_reference_compatible(he,inst,ids,ctp,device,args.cell_batch_size,args.preprocess_mode)
    if len(raw)!=len(meta):
        raise RuntimeError(f"reference feature/cardinality mismatch: raw={len(raw)} metadata={len(meta)}")
    started=time.perf_counter()
    with torch.inference_mode():
        padded=np.zeros((1,args.max_cells,768),np.float32); mask=np.zeros((1,args.max_cells),np.float32); padded[0,:len(raw)]=raw; mask[0,:len(raw)]=1
        _,reg,proj,_=xcell(raw_images=None,x=torch.from_numpy(padded).to(device),mask=torch.from_numpy(mask).to(device))
    profile.update({"read_sec":read_sec,"xcell_sec":time.perf_counter()-started})
    return {"patch_id":row["patch_id"],"wsi_id":row["wsi_id"],"total_cell_count":total,"source_index":int(row["source_index"]) if "source_index" in row else None,"cells":meta.tolist(),"reg_features":reg[0,:len(raw)].cpu().numpy().tolist(),"proj_features":proj[0,:len(raw)].cpu().numpy().tolist()},profile


def main() -> None:
    args=parse(); device=torch.device(args.device); cfg=load_config(args.config); stamp=args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"); out=Path(args.output_root)/f"xcell_features_{args.split}_{stamp}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    rows=pd.read_parquet(args.patch_index).query("split == @args.split").to_dict("records"); end=args.end if args.end is not None else len(rows)
    if args.start<0 or end<=args.start or end>len(rows): raise ValueError(f"invalid range {args.start}:{end} for {len(rows)} rows")
    rows=rows[args.start:end]; rows=rows[:args.limit] if args.limit else rows; out.mkdir(parents=True); pairs={pair.wsi_id:pair for pair in read_pairs(cfg)}
    missing=set(row["wsi_id"] for row in rows)-set(pairs)
    if missing: raise ValueError(f"missing nuclei pairs: {sorted(missing)[:5]}")
    ctp,xcell=load_models(args,device); refs=[]; slide_cache={}; timings=[]
    with (out/"completed.jsonl").open("a") as completed, (out/"failures.jsonl").open("a") as failures:
        for local_start in range(0,len(rows),args.shard_size):
            part=rows[local_start:local_start+args.shard_size]
            try:
                payload=[]
                for row in part:
                    wsi_id=row["wsi_id"]
                    if wsi_id not in slide_cache: slide_cache[wsi_id]=open_slide_images(pairs[wsi_id])
                    value,profile=extract_one(row,slide_cache[wsi_id],ctp,xcell,args,device); payload.append(value); profile["patch_id"]=value["patch_id"]; timings.append(profile)
            except Exception as exc:
                failures.write(json.dumps({"start":args.start+local_start,"error":repr(exc)})+"\n"); failures.flush(); raise
            start=args.start+local_start; end=start+len(part); shard=out/f"features_{start:07d}_{end-1:07d}.parquet"
            pq.write_table(pa.Table.from_pylist(payload,schema=FEATURE_SCHEMA),shard,compression="zstd")
            refs.append({"shard_path":str(shard),"start":start,"end":end,"rows":len(part)}); completed.write(json.dumps(refs[-1])+"\n"); completed.flush(); print({"event":"shard_complete","done":local_start+len(part),"total":len(rows)},flush=True)
    pd.DataFrame(refs).to_parquet(out/"feature_index.parquet",index=False); pd.DataFrame(timings).to_parquet(out/"timing_by_patch.parquet",index=False)
    summary={key:float(pd.DataFrame(timings)[key].mean()) for key in ("read_sec","preprocess_sec","attention_sec","transform_sec","ctrans_sec","xcell_sec")} if timings else {}
    (out/"metadata.json").write_text(json.dumps({"split":args.split,"rows":len(rows),"max_cells":args.max_cells,"preprocess_mode":args.preprocess_mode,"selection_policy":args.selection_policy,"spatial_grid_size":args.spatial_grid_size,"cell_features":"HE soft-attention CTransPath-768 -> XCellFormer reg/proj-64","ctranspath":args.ctranspath_checkpoint,"xcell":args.xcell_checkpoint,"mean_timing_seconds":summary},indent=2)); print({"event":"complete","output":str(out),"mean_timing_seconds":summary},flush=True)


if __name__=="__main__": main()
