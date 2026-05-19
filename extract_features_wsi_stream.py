import os
import argparse
import pickle
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import tifffile
import zarr
from PIL import Image
from tqdm import tqdm
import logging
import scipy.ndimage as ndimage
from utils import load_cellpose_model
from module.TransPath.ctran import ctranspath

# =========================
# 配置与日志
# =========================
def setup_logger(log_dir, process_id=None):
    os.makedirs(log_dir, exist_ok=True)
    name = "WSI_Stream" if process_id is None else f"WSI_Stream_P{process_id}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        log_file = "stream_extract.log" if process_id is None else f"stream_extract_p{process_id}.log"
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def is_patch_valid(he_patch, black_threshold=15, white_threshold=240, std_threshold=5.0, black_area_threshold=0.6):
    if he_patch.size == 0: return False
    mean_val = he_patch.mean()
    if mean_val < black_threshold or mean_val > white_threshold: return False
    if he_patch.std() < std_threshold: return False
    gray = he_patch.mean(axis=-1)
    black_ratio = np.sum(gray < black_threshold) / gray.size
    if black_ratio > black_area_threshold: return False
    return True

def adaptive_load_ctranspath(model, weight_path, device):
    """
    CTransPath 权重适配器 (timm 0.5.4 -> 0.9.x)
    """
    logger = logging.getLogger("WSI_Stream")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if "layers.0.downsample" in k:
            new_k = k.replace("layers.0.downsample", "layers.1.downsample")
        elif "layers.1.downsample" in k:
            new_k = k.replace("layers.1.downsample", "layers.2.downsample")
        elif "layers.2.downsample" in k:
            new_k = k.replace("layers.2.downsample", "layers.3.downsample")
        new_state_dict[new_k] = v
    
    from module.TransPath.ctran import ConvStem
    model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=model.embed_dim)
    
    model.load_state_dict(new_state_dict, strict=False)
    
    class GlobalPoolHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            if x.dim() == 4:
                x = x.permute(0, 3, 1, 2)
                x = self.pool(x)
            return torch.flatten(x, 1)

    model.head = GlobalPoolHead()
    model.to(device)
    return model

# =========================
# Worker Process
# =========================
def worker_process(process_id, coords_chunk, args, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.log_dir, process_id)
    
    logger.info(f"Worker {process_id} started on {device} with {len(coords_chunk)} patches.")
    
    # 1. 加载模型
    logger.info(f"[Worker {process_id}] Loading models on {device}...")
    cellpose_model = load_cellpose_model(model_type='/home/zyh/.cellpose/models/cpsam', device=device)
    
    ctp_model = ctranspath().to(device)
    ctp_model = adaptive_load_ctranspath(ctp_model, args.weights, device)
    ctp_model.eval()
    
    # 2. 打开 WSI
    he_z = zarr.open(tifffile.imread(args.he, aszarr=True), mode='r')['0']
    mif_z = zarr.open(tifffile.imread(args.mif, aszarr=True), mode='r')['0']
    num_mif_channels = mif_z.shape[0]
    
    he_cache_dir = os.path.join(args.cache_dir, "he")
    mif_cache_dir = os.path.join(args.cache_dir, "mif")
    he_img_dir = os.path.join(args.cache_dir, "he_images")
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    # tqdm 进度条，使用 position 使其不会互相覆盖
    pbar = tqdm(coords_chunk, desc=f"Worker {process_id} GPU {gpu_id}", position=process_id, leave=True)
    
    for x, y in pbar:
        he_pkl = os.path.join(he_cache_dir, f"he_x{x}_y{y}.pkl")
        mif_pkl = os.path.join(mif_cache_dir, f"mif_x{x}_y{y}.pkl")
        
        if os.path.exists(he_pkl) and os.path.exists(mif_pkl):
            continue
            
        he_patch = he_z[y:y+args.size, x:x+args.size, :]
        if not is_patch_valid(he_patch):
            continue
            
        # A. Cellpose 分割
        try:
            cp_results = cellpose_model.eval(he_patch, diameter=18, channels=[0, 0])
            masks = cp_results[0]
            unique_labels = np.unique(masks)
            unique_labels = unique_labels[unique_labels != 0]
            if len(unique_labels) == 0: continue
        except Exception as e:
            logger.error(f"[Worker {process_id}] Cellpose error at x={x}, y={y}: {e}")
            continue
        
        # B. 特征提取准备
        try:
            margin = 20
            centers = ndimage.center_of_mass(np.ones_like(masks), labels=masks, index=unique_labels)
            
            valid_labels = []
            valid_centers = []
            for label, (cy, cx) in zip(unique_labels, centers):
                if margin <= cx < args.size - margin and margin <= cy < args.size - margin:
                    valid_labels.append(label)
                    valid_centers.append((int(cy), int(cx)))
            
            if not valid_labels: continue
            valid_labels = valid_labels[:255] # 限制数量
            valid_centers = valid_centers[:255]
            
            # C. HE 特征提取 (广播机制生成掩膜)
            he_tensor = torch.from_numpy(he_patch).permute(2, 0, 1).float().to(device) / 255.0 # [3, H, W]
            masks_tensor = torch.from_numpy(masks).to(device) # [H, W]
            
            valid_labels_t = torch.tensor(valid_labels, device=device).view(-1, 1, 1) # [N, 1, 1]
            
            # 广播机制生成所有细胞的 mask: [N, H, W]
            cell_masks = (masks_tensor.unsqueeze(0) == valid_labels_t).float()
            
            # 矩阵相乘生成 masked patch: [N, 1, H, W] * [1, 3, H, W] -> [N, 3, H, W]
            masked_patches = cell_masks.unsqueeze(1) * he_tensor.unsqueeze(0)
            
            # 统一插值到模型输入尺寸
            input_batch = F.interpolate(masked_patches, size=(224, 224), mode='bilinear', align_corners=False)
            input_batch = (input_batch - mean) / std # 标准化
            
            # 批量推理
            cell_features_list = []
            with torch.no_grad():
                for i in range(0, input_batch.size(0), args.batch_size):
                    batch_out = ctp_model(input_batch[i : i + args.batch_size])
                    cell_features_list.append(batch_out.cpu().numpy())
            
            cell_features = np.concatenate(cell_features_list, axis=0)
            
            # 保存 HE PKL
            feat_arr = np.zeros((255, 768), dtype=np.float32)
            mask_arr = np.zeros(255, dtype=np.float32)
            n_cells = cell_features.shape[0]
            feat_arr[:n_cells] = cell_features
            mask_arr[:n_cells] = 1.0
            
            # 构造过滤后的 mask 数组
            filtered_masks = np.zeros_like(masks)
            for label in valid_labels:
                filtered_masks[masks == label] = label
                
            with open(he_pkl, "wb") as f:
                pickle.dump({"features": torch.from_numpy(feat_arr).unsqueeze(0), 
                             "mask": torch.from_numpy(mask_arr).unsqueeze(0), 
                             "cell_masks": filtered_masks}, f)
                             
            Image.fromarray(he_patch).save(os.path.join(he_img_dir, f"he_x{x}_y{y}.png"))
            
            # D. mIF 密度提取 (使用 cell mask 直接聚合，避免邻域污染)
            mif_patch = mif_z[:, y:y+args.size, x:x+args.size]
            mif_tensor = torch.from_numpy(mif_patch).float().unsqueeze(0).to(device) # [1, C, H, W]
            mif_tensor = mif_tensor * (mif_tensor > 0.5).float()
            
            # cell_masks [N, H, W] 复用自步骤 C
            # 矩阵乘法: [C, HW] @ [HW, N] -> [C, N], .T -> [N, C]
            n_cells = cell_masks.shape[0]
            cell_masks_flat = cell_masks.view(n_cells, -1)  # [N, HW]
            mif_flat = mif_tensor.squeeze(0).view(num_mif_channels, -1)  # [C, HW]
            pixel_counts = cell_masks_flat.sum(dim=1, keepdim=True) + 1e-8  # [N, 1]
            density = ((mif_flat @ cell_masks_flat.T).T / pixel_counts).cpu().numpy() / 255.0  # [N, C]
            
            mif_feat_arr = np.zeros((255, num_mif_channels), dtype=np.float32)
            mif_mask_arr = np.zeros(255, dtype=np.float32)
            n_mif = density.shape[0]
            mif_feat_arr[:n_mif] = density
            mif_mask_arr[:n_mif] = 1.0
            
            with open(mif_pkl, "wb") as f:
                pickle.dump({"features": torch.from_numpy(mif_feat_arr).unsqueeze(0), 
                             "mask": torch.from_numpy(mif_mask_arr).unsqueeze(0)}, f)
            
        except Exception as e:
            import traceback
            logger.error(f"[Worker {process_id}] Error at x={x}, y={y}: {e}")
            logger.error(traceback.format_exc())
            continue

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="WSI Stream Feature Extraction (Multi-GPU Optimized)")
    parser.add_argument("--he", type=str, required=True, help="Path to HE ome.tif")
    parser.add_argument("--mif", type=str, required=True, help="Path to mIF ome.tiff")
    parser.add_argument("--cache_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--size", type=int, default=512, help="Patch size")
    parser.add_argument("--weights", type=str, default="./module/checkpoint/ctranspath.pth", help="CTransPath weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Cell inference batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes (GPUs)")
    args = parser.parse_args()
    
    # 强制使用 spawn 方法启动多进程（CUDA 需要）
    mp.set_start_method('spawn', force=True)

    logger = setup_logger(args.log_dir)
    logger.info("Initializing multi-GPU streaming extraction...")
    
    os.makedirs(os.path.join(args.cache_dir, "he"), exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "mif"), exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "he_images"), exist_ok=True)
    
    logger.info("Reading WSI metadata...")
    he_z = zarr.open(tifffile.imread(args.he, aszarr=True), mode='r')['0']
    H, W = he_z.shape[0], he_z.shape[1]
    
    coords = []
    for y in range(0, H - args.size, args.size):
        for x in range(0, W - args.size, args.size):
            coords.append((x, y))
            
    logger.info(f"Total potential patches: {len(coords)}")
    
    if args.num_workers > 1:
        chunk_size = len(coords) // args.num_workers
        chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        # 处理余数
        if len(chunks) > args.num_workers:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
            
        processes = []
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("No GPUs detected! Using CPU.")
            num_gpus = 1
            
        for i in range(args.num_workers):
            gpu_id = i % num_gpus
            p = mp.Process(target=worker_process, args=(i, chunks[i], args, gpu_id))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        # 单进程
        worker_process(0, coords, args, 0)
        
    logger.info(f"All done! Features saved to {args.cache_dir}")

if __name__ == "__main__":
    main()
