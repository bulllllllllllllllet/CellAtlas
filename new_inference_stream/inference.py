"""
==============================================================================
XCellAligner 新训练流水线 — HE→mIF 对齐模型推理脚本
==============================================================================

【训练流水线回顾】（对应 new_training_stream/）
  1. compute_global_norm_stats.py   → 计算 mIF 全局 p99 归一化统计量
  2. extract_features_wsi_stream.py  → 多 GPU 流式提取 WSI 特征（HE + mIF）
  3. re_extract_crc02.py             → （可选）用 soft attention centered crop
                                       重提取 HE 特征，提升质量
  4. multidata_aligner_trainer.py    → 训练 XCellFormer 对齐模型

【推理流程】（本脚本）
  输入图像 (PNG/JPG/SVS/TIFF)
      │
      ├─ 小图（非 WSI）──→ process_single_image()
      │                       ├─ Cellpose 细胞分割 → masks
      │                       ├─ extract_he_features_soft_attention()
      │                       │    对每个细胞:
      │                       │      以质心为中心裁 128×128
      │                       │      高斯模糊 mask → soft attention 图
      │                       │      图像 × 注意力图 → CTransPath → 768-dim
      │                       ├─ Pad → [1, 255, 768]
      │                       └─ XCellFormer → reg_out + proj_out [1, 255, 64]
      │
      └─ 大图 / WSI ──────→ _process_large_image_by_patches()
                                ├─ 分块 2048×2048
                                ├─ 每块：Cellpose → soft_attention → CTransPath
                                │         → XCellFormer
                                └─ 合并 masks (label_offset) + 拼接 features
              │
              ▼
    batch_inference() 汇总
        ├─ 保存 reg_out   → features_{name}_reg.npy   (64-dim, MSE 对齐头)
        ├─ 保存 proj_out  → features_{name}_proj.npy   (64-dim, 对比学习头)
        ├─ 保存 masks     → masks_{name}.npy
        └─ KMeans 聚类（基于 reg_out）
            ├─ {name}_cluster.png    降采样可视化
            └─ all_cells_info.json   (质心坐标 + 聚类标签)

【模型配置】
  - XCellFormer: input_dim=768, hidden_dim=512, n_heads=8, num_layers=4
                  output_dim=64, max_cells=255, use_large_vit=False
  - CTransPath:   timm 兼容加载 (layers.0/1/2.downsample 键名重映射)
  - Cellpose:     diameter=18, 模式 cpsam

【输出说明】
  - reg_out（regression head）:   被 mIF 直接 MSE 监督，推荐用于下游任务
  - proj_out（projection head）:  对比学习监督，语义一致性更强
  - 聚类默认使用 reg_out，两种特征均保存供用户选择
==============================================================================
"""

import os
import sys
import json
import argparse
import threading
import numpy as np
from PIL import Image
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans, MiniBatchKMeans
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import openslide

Image.MAX_IMAGE_PIXELS = None

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from torchvision import transforms

from utils import load_cellpose_model
from XCellFormer import XCellFormer

try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")


# =========================
# 全局锁（多线程 GPU 互斥）
# =========================
_gpu_lock = threading.Lock()


# =========================
# Soft Attention Centered Crop 参数
# （来自 new_training_stream/re_extract_crc02.py）
# =========================
CROP_SIZE = 128
HALF_CROP = CROP_SIZE // 2
GAUSS_SIGMA = 12.0
BG_WEIGHT = 0.35
FG_WEIGHT = 0.65


# =========================
# Soft Attention 辅助函数
# =========================

def crop_region(he_patch_np, cy, cx):
    """
    以细胞质心 (cy, cx) 为中心，从 HE 图像上裁剪 128×128 区域。
    若靠近边缘，用 edge padding 填充至固定大小。
    """
    h, w = he_patch_np.shape[:2]
    y_start = max(0, cy - HALF_CROP)
    y_end = min(h, cy + HALF_CROP)
    x_start = max(0, cx - HALF_CROP)
    x_end = min(w, cx + HALF_CROP)
    region = he_patch_np[y_start:y_end, x_start:x_end]
    pad_top = max(0, HALF_CROP - cy)
    pad_bottom = max(0, cy + HALF_CROP - h)
    pad_left = max(0, HALF_CROP - cx)
    pad_right = max(0, cx + HALF_CROP - w)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        region = np.pad(region,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='edge')
    return region


def crop_region_mask(mask, cy, cx):
    """
    以细胞质心 (cy, cx) 为中心，从 cell mask 上裁剪 128×128 区域。
    与 crop_region 完全相同的裁剪 + padding 逻辑，用于 cell_masks。
    """
    h, w = mask.shape[:2]
    y_start = max(0, cy - HALF_CROP)
    y_end = min(h, cy + HALF_CROP)
    x_start = max(0, cx - HALF_CROP)
    x_end = min(w, cx + HALF_CROP)
    region = mask[y_start:y_end, x_start:x_end]
    pad_top = max(0, HALF_CROP - cy)
    pad_bottom = max(0, cy + HALF_CROP - h)
    pad_left = max(0, HALF_CROP - cx)
    pad_right = max(0, cx + HALF_CROP - w)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        region = np.pad(region,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='edge')
    return region


def apply_soft_attention(image_crop, label_mask):
    """
    对裁剪区域应用 soft attention：
      1. 对 label_mask（当前细胞 mask）做高斯模糊
      2. 构造注意力图：背景权重 0.35 + 前景权重 0.65 × (归一化模糊图)
      3. 图像 × 注意力图（保留上下文，突出目标细胞）
    """
    if label_mask.sum() == 0:
        return image_crop.astype(np.float32) / 255.0
    blurred = ndimage.gaussian_filter(label_mask, sigma=GAUSS_SIGMA, mode='nearest')
    attn_map = BG_WEIGHT + FG_WEIGHT * (blurred / (blurred.max() + 1e-6))
    image_float = image_crop.astype(np.float32) / 255.0
    return image_float * attn_map[..., None]


# =========================
# 模型加载
# =========================

def adaptive_load_ctranspath(model, weight_path, device):
    """
    CTransPath 权重适配器（兼容 timm 0.5.4 → 0.9.x 的键名差异）。
    将 layers.0/1/2.downsample 重映射为 layers.1/2/3.downsample，
    并使用 ConvStem 替换 patch_embed。
    """
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


def load_ctranspath(weight_path, device):
    """加载 CTransPath 模型（实例化 + 权重适配 + eval 模式）。"""
    model = ctranspath()
    model = adaptive_load_ctranspath(model, weight_path, device)
    model.eval()
    return model


def load_xcell_model(checkpoint_path, device):
    """
    加载 XCellFormer 对齐模型。
    配置与 new_training_stream/multidata_aligner_trainer.py 完全一致：
      - input_dim=768 (CTransPath 输出)
      - hidden_dim=512, n_heads=8, num_layers=4
      - output_dim=64 (reg + proj 双头)
      - max_cells=255, use_large_vit=False
    """
    model = XCellFormer(
        input_dim=768,
        hidden_dim=512,
        n_heads=8,
        num_layers=4,
        output_dim=64,
        max_cells=255,
        use_large_vit=False,
        device=str(device),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# =========================
# HE 特征提取 — Soft Attention Centered Crop
# =========================

def extract_he_features_soft_attention(he_patch, cell_masks, ctp_model, device, batch_size=64, max_cells=255):
    """
    核心特征提取函数。
    与 new_training_stream/re_extract_crc02.py 完全一致的流程：
      1. 从 Cellpose mask 中提取每个细胞的 centroid
      2. 以 centroid 为中心裁 128×128 区域（边缘用 edge padding）
      3. 对 mask 区域做高斯模糊 → 构造 soft attention 图
      4. 图像 × 注意力图（保留 35% 背景 + 增强 65% 前景）
      5. Resize 到 224×224 + ImageNet 归一化
      6. 批量送入 CTransPath → [N, 768]

    与旧流水线（CellEngine）的关键区别：不使用形态学调制（面积/周长/圆度）。
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    he_patch = np.asarray(he_patch, dtype=np.uint8)
    if he_patch.ndim == 2:
        he_patch = np.stack([he_patch] * 3, axis=-1)
    elif he_patch.shape[2] > 3:
        he_patch = he_patch[:, :, :3]

    cell_masks = np.asarray(cell_masks, dtype=np.int32)
    unique_labels = np.unique(cell_masks)
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) == 0:
        return None

    unique_labels = unique_labels[:max_cells]
    centers = ndimage.center_of_mass(np.ones_like(cell_masks), labels=cell_masks, index=unique_labels)
    cells = sorted(zip(unique_labels, centers), key=lambda x: (x[1][0], x[1][1]))
    valid_labels = [c[0] for c in cells]
    valid_centers = [c[1] for c in cells]

    crops = []
    for label, (cy, cx) in zip(valid_labels, valid_centers):
        cy, cx = int(cy), int(cx)
        crop_128 = crop_region(he_patch, cy, cx)
        crop_mask = crop_region_mask(cell_masks, cy, cx)
        label_mask = (crop_mask == label).astype(np.float32)
        attended = apply_soft_attention(crop_128, label_mask)
        attended_uint8 = (np.clip(attended, 0, 1) * 255).astype(np.uint8)
        crops.append(preprocess(Image.fromarray(attended_uint8)))

    if not crops:
        return None

    cell_features = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i + batch_size]).to(device)
        with torch.no_grad():
            out = ctp_model(batch)
        cell_features.append(out.cpu().numpy())

    return np.concatenate(cell_features, axis=0)


# =========================
# 数据处理工具
# =========================

def pad_features_to_max_cells(features, max_cells=255):
    """
    将细胞特征填充/截断到固定长度 255。
    返回 (padded_array, mask)，mask 中 1.0 表示有效细胞，0.0 表示 padding。
    与训练流水线的 max_cells=255 一致。
    """
    if features is None:
        return None, None
    num_cells, feature_dim = features.shape
    n = min(num_cells, max_cells)
    padded = np.zeros((max_cells, feature_dim), dtype=np.float32)
    mask = np.zeros(max_cells, dtype=np.float32)
    padded[:n] = features[:n]
    mask[:n] = 1.0
    return padded, mask


def save_features_to_disk(features, image_name, output_dir, suffix=""):
    """
    将特征保存为 .npy 文件（使用 memmap 分块写入，避免大文件 OOM）。
    suffix 用于区分 reg_out (_reg) 和 proj_out (_proj)。
    """
    if features is None:
        return None
    features_file = os.path.join(output_dir, f"features_{os.path.splitext(image_name)[0]}{suffix}.npy")
    if isinstance(features, np.ndarray):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        num_cells = features.shape[0]
        feature_dim = features.shape[1]
        memmap = np.lib.format.open_memmap(
            features_file, mode="w+", dtype=features.dtype, shape=(num_cells, feature_dim)
        )
        chunk_size = 100000
        for start in range(0, num_cells, chunk_size):
            end = min(start + chunk_size, num_cells)
            memmap[start:end] = features[start:end]
        del memmap
    return features_file


def save_masks_to_disk(masks, image_name, output_dir):
    """将细胞分割掩码保存为 .npy 文件。"""
    masks_file = os.path.join(output_dir, f"masks_{os.path.splitext(image_name)[0]}.npy")
    np.save(masks_file, masks)
    return masks_file


def get_cell_centroids(masks):
    """
    极速版质心计算（专为 WSI 大图优化，避免 OOM）。
    利用 bincount 替代 for 循环 + np.unique，复杂度 O(N)。
    """
    if masks is None:
        return []
    masks = np.asarray(masks)
    if masks.ndim != 2:
        masks = masks.squeeze()
    ys, xs = np.nonzero(masks)
    if ys.size == 0:
        return []
    labels = masks[ys, xs].astype(np.int64, copy=False)
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    sum_y = np.bincount(labels, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(labels, weights=xs, minlength=max_label + 1)
    valid_labels = np.where(counts > 0)[0]
    valid_labels = valid_labels[valid_labels > 0]
    if valid_labels.size == 0:
        return []
    centroid_x = (sum_x[valid_labels] / counts[valid_labels]).astype(np.int64)
    centroid_y = (sum_y[valid_labels] / counts[valid_labels]).astype(np.int64)
    return list(zip(centroid_x.tolist(), centroid_y.tolist()))


def visualize_clusters(masks, cluster_labels, save_path, k):
    """
    优化版聚类可视化：
      1. 降采样到最长边 ≤ 4096
      2. LUT 映射：细胞 label → 聚类标签 → 颜色
      3. 使用 tab10 colormap 渲染
    """
    masks = np.asarray(masks)
    if masks.ndim != 2:
        masks = masks.squeeze()
    h, w = masks.shape
    max_visual_side = 4096
    stride = max(1, int(np.ceil(max(h, w) / max_visual_side)))
    masks_view = masks[::stride, ::stride].copy()
    num_cells = len(cluster_labels)
    max_label = int(masks_view.max()) if masks_view.size > 0 else 0
    lut_size = max(num_cells + 1, max_label + 1)
    label_to_cluster = np.zeros(lut_size, dtype=np.int16)
    label_to_cluster[1:num_cells + 1] = np.asarray(cluster_labels, dtype=np.int16) + 1
    cluster_map = label_to_cluster[masks_view]
    colors = plt.cm.get_cmap('tab10', k)
    color_lut = np.zeros((k + 1, 3), dtype=np.uint8)
    for idx in range(1, k + 1):
        color_lut[idx] = (np.array(colors(idx - 1)[:3]) * 255).astype(np.uint8)
    result_img = color_lut[cluster_map]
    result_pil = Image.fromarray(result_img)
    result_pil.save(save_path)


def is_patch_valid(he_patch, black_threshold=15, white_threshold=240, std_threshold=5.0, black_area_threshold=0.6):
    """
    过滤空白/无效 patch：
      - 均值过低（全黑）或过高（全白）→ 无效
      - 标准差过低（纯色背景）→ 无效
      - 黑色像素占比过高 → 无效
    """
    if he_patch.size == 0:
        return False
    mean_val = he_patch.mean()
    if mean_val < black_threshold or mean_val > white_threshold:
        return False
    if he_patch.std() < std_threshold:
        return False
    gray = he_patch.mean(axis=-1)
    black_ratio = np.sum(gray < black_threshold) / gray.size
    if black_ratio > black_area_threshold:
        return False
    return True


# =========================
# 单图推理入口
# =========================

def process_single_image(image_path, ctp_model, xcell_model, cellpose_model, device, batch_size=64):
    """
    处理单张图像的完整推理流程。
    - 小图（<300MB）直接送入 Cellpose → Soft Attention CTransPath → XCellFormer
    - 大图/WSI 自动切换为分块模式（_process_large_image_by_patches）

    返回: (masks, reg_features, proj_features, image_name, image_path)
      masks: [H, W] int32 Cellpose 分割掩码
      reg_features: [N, 64] XCellFormer regression head 输出
      proj_features: [N, 64] XCellFormer projection head 输出
    """
    use_tiling = False
    if isinstance(image_path, str) and os.path.isfile(image_path):
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".svs", ".tif", ".tiff"]:
            use_tiling = True
        else:
            file_size = os.path.getsize(image_path) / (1024 * 1024)
            if file_size > 300:
                use_tiling = True

    if use_tiling:
        tile_workers = min(8, max(1, os.cpu_count() or 1))
        result = _process_large_image_by_patches(
            image_path, ctp_model, xcell_model, cellpose_model, device,
            batch_size=batch_size, max_workers=tile_workers
        )
        if result is None:
            return None
        masks, reg_features, proj_features = result
        return masks, reg_features, proj_features, os.path.basename(image_path), image_path

    with _gpu_lock:
        if isinstance(image_path, str):
            img_np = np.array(Image.open(image_path).convert("RGB"))
        else:
            img_np = image_path
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        masks, flows, styles = cellpose_model.eval(img_np, diameter=18, channels=[0, 0])

    cell_features = extract_he_features_soft_attention(img_np, masks, ctp_model, device, batch_size)
    if cell_features is None:
        return None

    raw_768, raw_mask = pad_features_to_max_cells(cell_features, max_cells=255)
    if raw_768 is None:
        return None

    x_tensor = torch.from_numpy(raw_768).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        _, reg_out, proj_out, _ = xcell_model(raw_images=None, x=x_tensor, mask=mask_tensor)

    n_valid = min(len(cell_features), 255)
    reg_features = reg_out[0, :n_valid].cpu().numpy()
    proj_features = proj_out[0, :n_valid].cpu().numpy()

    return masks, reg_features, proj_features, os.path.basename(image_path), image_path


# =========================
# WSI 分块推理
# =========================

def _process_large_image_by_patches(image_path, ctp_model, xcell_model, cellpose_model, device,
                                     tile_size=2048, overlap=0, max_workers=4, batch_size=64):
    """
    WSI 大图分块推理：
      1. 将 WSI 划分为 tile_size×tile_size 的块（步长 tile_size - overlap）
      2. 多线程读取 + GPU 互斥推理（_gpu_lock）
      3. 每块独立完成 Cellpose → soft attention → CTransPath → XCellFormer
      4. 使用 label_offset 合并 masks 避免细胞 label 冲突
      5. 返回 (full_mask, all_reg_features, all_proj_features)

    线程安全设计：
      - 每个线程持有一个独立的 openslide 句柄（thread_local）
      - GPU 推理通过 _gpu_lock 互斥
    """
    slide = openslide.OpenSlide(image_path)
    width, height = slide.level_dimensions[0]
    slide.close()

    step_size = tile_size - overlap
    full_mask = np.zeros((height, width), dtype=np.int32)
    all_reg_features = []
    all_proj_features = []
    label_offset = 0

    tiles_x = (width + step_size - 1) // step_size
    tiles_y = (height + step_size - 1) // step_size
    total_tiles = tiles_x * tiles_y
    tile_tasks = []
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            tile_tasks.append((x, y, w, h))

    thread_local = threading.local()
    opened_slides = []
    opened_slides_lock = threading.Lock()

    def _get_thread_slide():
        local_slide = getattr(thread_local, "slide", None)
        if local_slide is None:
            local_slide = openslide.OpenSlide(image_path)
            thread_local.slide = local_slide
            with opened_slides_lock:
                opened_slides.append(local_slide)
        return local_slide

    def _run_tile(task):
        x, y, w, h = task
        try:
            local_slide = _get_thread_slide()
            tile_img = local_slide.read_region((x, y), 0, (w, h)).convert("RGB")
            tile_np = np.array(tile_img)
            if tile_np.dtype != np.uint8:
                tile_np = (tile_np * 255).astype(np.uint8) if tile_np.max() <= 1.0 else tile_np.astype(np.uint8)

            if not is_patch_valid(tile_np):
                return x, y, w, h, None, None, None, None

            with _gpu_lock:
                tile_masks, flows, styles = cellpose_model.eval(tile_np, diameter=18, channels=[0, 0])

            cell_feats = extract_he_features_soft_attention(tile_np, tile_masks, ctp_model, device, batch_size)
            if cell_feats is None:
                return x, y, w, h, tile_masks, None, None, None

            raw_768, raw_mask = pad_features_to_max_cells(cell_feats, max_cells=255)
            if raw_768 is None:
                return x, y, w, h, tile_masks, None, None, None

            x_t = torch.from_numpy(raw_768).unsqueeze(0).to(device)
            m_t = torch.from_numpy(raw_mask).unsqueeze(0).to(device)
            with torch.no_grad():
                _, reg_tile, proj_tile, _ = xcell_model(raw_images=None, x=x_t, mask=m_t)

            n = min(len(cell_feats), 255)
            reg_feats = reg_tile[0, :n].cpu().numpy()
            proj_feats = proj_tile[0, :n].cpu().numpy()

            return x, y, w, h, tile_masks, reg_feats, proj_feats, None

        except Exception as e:
            return x, y, w, h, None, None, None, str(e)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_tile, task) for task in tile_tasks]
            processed = 0
            for future in futures:
                x, y, w, h, tile_masks, reg_feats, proj_feats, err = future.result()
                processed += 1
                if processed % 10 == 0 or processed == total_tiles:
                    print(f"[TiledInfer] 分块进度: {processed}/{total_tiles}")
                if err is not None:
                    continue
                if tile_masks is None or reg_feats is None:
                    tile_masks_np = np.asarray(tile_masks) if tile_masks is not None else None
                    if tile_masks_np is not None and tile_masks_np.max() > 0:
                        tile_masks_np = tile_masks_np.astype(np.int32).copy()
                        tile_masks_np[tile_masks_np > 0] += label_offset
                        label_offset = int(tile_masks_np.max())
                        region = full_mask[y:y + h, x:x + w]
                        if region.shape != tile_masks_np.shape:
                            tile_masks_np = tile_masks_np[:region.shape[0], :region.shape[1]]
                        mask_new = tile_masks_np > 0
                        region_mask = (region == 0) & mask_new
                        region[region_mask] = tile_masks_np[region_mask]
                        full_mask[y:y + h, x:x + w] = region
                    continue

                all_reg_features.append(reg_feats)
                all_proj_features.append(proj_feats)

                tile_masks_np = np.asarray(tile_masks).astype(np.int32).copy()
                if tile_masks_np.max() > 0:
                    tile_masks_np[tile_masks_np > 0] += label_offset
                    label_offset = int(tile_masks_np.max())
                region = full_mask[y:y + h, x:x + w]
                if region.shape != tile_masks_np.shape:
                    tile_masks_np = tile_masks_np[:region.shape[0], :region.shape[1]]
                mask_new = tile_masks_np > 0
                region_mask = (region == 0) & mask_new
                region[region_mask] = tile_masks_np[region_mask]
                full_mask[y:y + h, x:x + w] = region

    finally:
        for s in opened_slides:
            try:
                s.close()
            except Exception:
                pass

    if len(all_reg_features) == 0:
        return None
    reg_all = np.concatenate(all_reg_features, axis=0)
    proj_all = np.concatenate(all_proj_features, axis=0)
    return full_mask, reg_all, proj_all


# =========================
# 批量推理 + 聚类
# =========================

def batch_inference(input_folder, output_folder, ctp_model, xcell_model, cellpose_model, device,
                    k=5, max_workers=2, batch_size=64):
    """
    批量推理入口：
      1. 扫描输入文件夹（支持 JPG/PNG/SVS/TIFF）
      2. 线程池并行处理每张图像（process_single_image）
      3. 保存 reg_out / proj_out / masks 到临时目录
      4. 加载所有 reg_out 特征进行跨图像 KMeans 聚类
      5. 生成聚类可视化（cluster.png）和全部细胞信息（all_cells_info.json）

    输出目录结构：
      output_folder/
      ├── temp_reg/       features_*_reg.npy
      ├── temp_proj/      features_*_proj.npy
      ├── temp_masks/     masks_*.npy
      ├── {name}_cluster.png
      └── all_cells_info.json
    """
    os.makedirs(output_folder, exist_ok=True)
    temp_reg_dir = os.path.join(output_folder, "temp_reg")
    temp_proj_dir = os.path.join(output_folder, "temp_proj")
    temp_masks_dir = os.path.join(output_folder, "temp_masks")
    os.makedirs(temp_reg_dir, exist_ok=True)
    os.makedirs(temp_proj_dir, exist_ok=True)
    os.makedirs(temp_masks_dir, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.svs']
    image_files = []
    if os.path.isfile(input_folder):
        if any(input_folder.lower().endswith(ext[1:]) for ext in image_extensions):
            image_files = [input_folder]
    else:
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    if not image_files:
        print(f"[Batch] 在 {input_folder} 中没有找到图像文件")
        return

    print(f"[Batch] 找到 {len(image_files)} 个图像文件")

    processed_images = []
    image_paths = []
    image_mask_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {}
        for i, img_path in enumerate(image_files):
            print(f"[Batch] [{i+1}/{len(image_files)}] 准备处理: {os.path.basename(img_path)}")
            future = executor.submit(process_single_image, img_path, ctp_model, xcell_model,
                                     cellpose_model, device, batch_size)
            future_to_image[future] = img_path

        completed = 0
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                result = future.result(timeout=600)
                if result:
                    masks, reg_feats, proj_feats, image_name, orig_path = result
                    base_name = os.path.splitext(image_name)[0]

                    save_features_to_disk(reg_feats, image_name, temp_reg_dir, suffix="_reg")
                    save_features_to_disk(proj_feats, image_name, temp_proj_dir, suffix="_proj")
                    saved_mask = save_masks_to_disk(masks, image_name, temp_masks_dir)

                    processed_images.append(image_name)
                    image_paths.append(orig_path)
                    image_mask_files.append(saved_mask)

                    completed += 1
                    print(f"[Batch] [{completed}/{len(image_files)}] {image_name} 完成")
                else:
                    completed += 1
                    print(f"[Batch] [{completed}/{len(image_files)}] 跳过 {os.path.basename(img_path)}")
            except Exception as e:
                completed += 1
                print(f"[Batch] [{completed}/{len(image_files)}] 处理 {os.path.basename(img_path)} 出错: {e}")

    if len(processed_images) == 0:
        print("[Batch] 没有成功处理的图像")
        return

    print("[Batch] 加载特征进行聚类...")
    reg_feature_files = sorted([
        os.path.join(temp_reg_dir, f) for f in os.listdir(temp_reg_dir)
        if f.startswith("features_") and f.endswith("_reg.npy")
    ])

    all_reg_features = []
    for ff in reg_feature_files:
        feats = np.load(ff)
        if feats.ndim == 2 and feats.shape[1] == 64:
            all_reg_features.append(feats)
        else:
            print(f"  [Skip] {ff}: shape={feats.shape}")

    if len(all_reg_features) == 0:
        print("[Batch] 无有效特征用于聚类")
        return

    all_reg_features = np.concatenate(all_reg_features, axis=0)
    print(f"[Batch] 聚类特征总细胞数: {len(all_reg_features)}")

    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
        if len(all_reg_features) >= 100000:
            print("[Batch] 细胞数 > 10万，使用 MiniBatchKMeans")
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=8192)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(all_reg_features)

    cluster_idx = 0
    all_cell_info = []
    for i, image_name in enumerate(processed_images):
        base_name = os.path.splitext(image_name)[0]
        reg_file = os.path.join(temp_reg_dir, f"features_{base_name}_reg.npy")
        if not os.path.exists(reg_file):
            continue
        feats = np.load(reg_file)
        current_clusters = clusters[cluster_idx:cluster_idx + len(feats)]
        cluster_idx += len(feats)

        masks_file = image_mask_files[i]
        if not os.path.exists(masks_file):
            continue
        masks = np.load(masks_file, mmap_mode='r')

        centroids = get_cell_centroids(masks)
        for j, (centroid, clabel) in enumerate(zip(centroids, current_clusters)):
            all_cell_info.append({
                "image_name": image_name,
                "cell_id": j,
                "centroid_x": int(centroid[0]),
                "centroid_y": int(centroid[1]),
                "cluster": int(clabel),
            })

        save_path = os.path.join(output_folder, f"{base_name}_cluster.png")
        visualize_clusters(masks, current_clusters, save_path, k)
        del masks

    all_cell_info_path = os.path.join(output_folder, "all_cells_info.json")
    with open(all_cell_info_path, "w", encoding="utf-8") as f:
        json.dump(all_cell_info, f, ensure_ascii=False, indent=2)
    print(f"[Batch] 聚类结果保存到: {all_cell_info_path}")
    print("[Batch] 完成！")


def main():
    """
    命令行入口。
    加载三个模型（Cellpose + CTransPath + XCellFormer），然后启动批量推理。
    """
    parser = argparse.ArgumentParser(description="XCellAligner 新训练流水线推理脚本")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output_folder", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, required=True, help="XCellFormer 权重 (.pth)")
    parser.add_argument("--ctranspath_checkpoint", type=str,
                        default="/home/zyh/NewMedLabel/XCellAligner/module/checkpoint/ctranspath.pth",
                        help="CTransPath 权重路径")
    parser.add_argument("--cellpose_model", type=str,
                        default="/home/zyh/.cellpose/models/cpsam",
                        help="Cellpose 模型路径")
    parser.add_argument("--k", type=int, default=7, help="聚类数")
    parser.add_argument("--max_workers", type=int, default=4, help="最大工作线程数")
    parser.add_argument("--batch_size", type=int, default=64, help="细胞特征推理批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("[1/4] 加载 Cellpose 分割模型...")
    cellpose_model = load_cellpose_model(model_type=args.cellpose_model, device=device)

    print("[2/4] 加载 CTransPath...")
    ctp_model = load_ctranspath(args.ctranspath_checkpoint, device)

    print("[3/4] 加载 XCellFormer...")
    xcell_model = load_xcell_model(args.model_path, device)

    print("[4/4] 开始批量推理...")
    batch_inference(args.input_folder, args.output_folder, ctp_model, xcell_model, cellpose_model,
                    device, k=args.k, max_workers=args.max_workers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
