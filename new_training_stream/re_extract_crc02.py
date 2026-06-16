"""
以 cell centroid 为中心裁区域，soft attention 强调目标细胞但保留上下文。
对比旧版：硬清零黑背景 → 上下文保留 + 目标细胞突出
"""
import os, sys, re, pickle, argparse, numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage import io
import tifffile
import zarr

import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CROP_SIZE = 128
RESIZE_SIZE = 224
HALF_CROP = CROP_SIZE // 2
GAUSS_SIGMA = 12.0
BG_WEIGHT = 0.35   # 背景保留程度
FG_WEIGHT = 0.65   # 前景放大程度
MAX_CELLS = 255
BATCH_SIZE = 64

def parse_xy(filename):
    m = re.search(r"x(\d+)_y(\d+)", filename)
    if m is None:
        raise ValueError(f"Invalid filename: {filename}")
    return int(m.group(1)), int(m.group(2))

def adaptive_load_ctranspath(model, weight_path, device):
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
    from module.TransPath.ctran import ctranspath
    model = ctranspath()
    model = adaptive_load_ctranspath(model, weight_path, device)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=Image.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def crop_region(he_patch_np, cy, cx):
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
    """和 crop_region 完全相同的裁剪+padding逻辑，用于 cell_masks"""
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
    return region  # [CROP_SIZE, CROP_SIZE]

def apply_soft_attention(image_crop, label_mask):
    """
    image_crop: [H, W, 3] uint8
    label_mask: [H, W] float32, 当前细胞对应的 mask (已crop+padding)
    """
    if label_mask.sum() == 0:
        return image_crop.astype(np.float32) / 255.0
    blurred = ndimage.gaussian_filter(label_mask, sigma=GAUSS_SIGMA, mode='nearest')
    attn_map = BG_WEIGHT + FG_WEIGHT * (blurred / (blurred.max() + 1e-6))
    image_float = image_crop.astype(np.float32) / 255.0
    return image_float * attn_map[..., None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_patches", type=int, default=None,
                        help="限制处理 patch 数量（用于 sanity check）")
    parser.add_argument("--mif_path", type=str, default=None,
                        help="mIF ome.tiff 路径（提供后重新抽取 mIF 而非拷贝旧缓存）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    he_out = os.path.join(args.output_dir, "he")
    mif_out = os.path.join(args.output_dir, "mif")
    img_out = os.path.join(args.output_dir, "he_images")
    os.makedirs(he_out, exist_ok=True)
    os.makedirs(mif_out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    print("[1/5] Loading CTransPath...")
    ctp_model = load_ctranspath(args.weights, device)

    if args.mif_path:
        print("[1.5/5] Loading mIF TIFF...")
        mif_z = zarr.open(tifffile.imread(args.mif_path, aszarr=True), mode='r')['0']
        num_mif_channels = mif_z.shape[0]
        print(f"       mIF channels: {num_mif_channels}")
    else:
        mif_z = None
        num_mif_channels = 0

    he_img_dir = os.path.join(args.input_dir, "he_images")
    he_pkl_dir = os.path.join(args.input_dir, "he")
    mif_pkl_dir = os.path.join(args.input_dir, "mif")

    he_files = sorted([f for f in os.listdir(he_img_dir) if f.endswith(".png")])
    if args.max_patches:
        he_files = he_files[:args.max_patches]
    print(f"[2/5] Found {len(he_files)} patches")

    print("[3/5] Re-extracting HE features (soft attention centered crop)...")
    for fname in tqdm(he_files):
        x, y = parse_xy(fname)
        he_img_path = os.path.join(he_img_dir, fname)
        he_pkl_path = os.path.join(he_pkl_dir, f"he_x{x}_y{y}.pkl")
        mif_pkl_path = os.path.join(mif_pkl_dir, f"mif_x{x}_y{y}.pkl")

        new_he_pkl = os.path.join(he_out, f"he_x{x}_y{y}.pkl")
        new_mif_pkl = os.path.join(mif_out, f"mif_x{x}_y{y}.pkl")
        new_img_path = os.path.join(img_out, fname)

        if os.path.exists(new_he_pkl) and os.path.exists(new_mif_pkl):
            continue

        try:
            he_patch = io.imread(he_img_path)
        except:
            he_patch = np.array(Image.open(he_img_path).convert("RGB"))
        if he_patch.ndim == 2:
            he_patch = np.stack([he_patch]*3, axis=-1)
        elif he_patch.shape[2] > 3:
            he_patch = he_patch[:, :, :3]

        if not os.path.exists(he_pkl_path):
            continue
        with open(he_pkl_path, "rb") as f:
            old_data = pickle.load(f)
        cell_masks = old_data["cell_masks"]

        unique_labels = np.unique(cell_masks)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            continue

        # 计算 centroid 并按空间顺序排列
        centers = ndimage.center_of_mass(
            np.ones_like(cell_masks), labels=cell_masks, index=unique_labels
        )
        cells = sorted(zip(unique_labels, centers),
                       key=lambda x: (x[1][0], x[1][1]))
        cells = cells[:MAX_CELLS]
        valid_labels = [c[0] for c in cells]
        valid_centers = [c[1] for c in cells]

        # 为每个细胞：crop + soft attention + 预处理
        crops = []
        for label, (cy, cx) in zip(valid_labels, valid_centers):
            cy, cx = int(cy), int(cx)
            crop_128 = crop_region(he_patch, cy, cx)
            crop_mask = crop_region_mask(cell_masks, cy, cx)
            label_mask = (crop_mask == label).astype(np.float32)
            attended = apply_soft_attention(crop_128, label_mask)
            attended_uint8 = (np.clip(attended, 0, 1) * 255).astype(np.uint8)
            crop_pil = Image.fromarray(attended_uint8)
            crops.append(preprocess(crop_pil))

        if not crops:
            continue

        cell_features = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = torch.stack(crops[i:i+BATCH_SIZE]).to(device)
            with torch.no_grad():
                out = ctp_model(batch)
            cell_features.append(out.cpu().numpy())
        cell_features = np.concatenate(cell_features, axis=0)

        n = min(len(cell_features), MAX_CELLS)
        feat_arr = np.zeros((MAX_CELLS, 768), dtype=np.float32)
        mask_arr = np.zeros(MAX_CELLS, dtype=np.float32)
        feat_arr[:n] = cell_features[:n]
        mask_arr[:n] = 1.0

        with open(new_he_pkl, "wb") as f:
            pickle.dump({
                "features": torch.from_numpy(feat_arr).unsqueeze(0),
                "mask": torch.from_numpy(mask_arr).unsqueeze(0),
                "cell_masks": cell_masks,
                "centers": [(float(cy), float(cx)) for (cy, cx) in valid_centers],
            }, f)

        if not os.path.exists(new_mif_pkl):
            if args.mif_path:
                patch_h, patch_w = cell_masks.shape[:2]
                mif_patch = mif_z[:, y:y+patch_h, x:x+patch_w]
                mif_tensor = torch.from_numpy(mif_patch).float().unsqueeze(0).to(device)
                mif_tensor = mif_tensor * (mif_tensor > 0.5).float()

                labels_t = torch.tensor(valid_labels, device=device).view(-1, 1, 1)
                masks_tensor = torch.from_numpy(cell_masks).to(device)
                bool_masks = (masks_tensor.unsqueeze(0) == labels_t).float()

                n_cells = bool_masks.shape[0]
                cell_flat = bool_masks.view(n_cells, -1)
                mif_flat = mif_tensor.squeeze(0).view(num_mif_channels, -1)
                pixel_counts = cell_flat.sum(dim=1, keepdim=True) + 1e-8
                density = ((mif_flat @ cell_flat.T).T / pixel_counts).cpu().numpy() / 255.0

                mif_feat_arr = np.zeros((MAX_CELLS, num_mif_channels), dtype=np.float32)
                mif_mask_arr = np.zeros(MAX_CELLS, dtype=np.float32)
                n = min(density.shape[0], MAX_CELLS)
                mif_feat_arr[:n] = density[:n]
                mif_mask_arr[:n] = 1.0

                with open(new_mif_pkl, "wb") as f:
                    pickle.dump({
                        "features": torch.from_numpy(mif_feat_arr).unsqueeze(0),
                        "mask": torch.from_numpy(mif_mask_arr).unsqueeze(0),
                    }, f)
            elif os.path.exists(mif_pkl_path):
                with open(mif_pkl_path, "rb") as f:
                    mif_data = pickle.load(f)
                with open(new_mif_pkl, "wb") as f:
                    pickle.dump(mif_data, f)

        if not os.path.exists(new_img_path):
            Image.fromarray(he_patch).save(new_img_path)

    print("[4/5] Copying global_norm_stats.json...")
    stats_src = os.path.join(args.input_dir, "global_norm_stats.json")
    if os.path.exists(stats_src):
        import shutil
        shutil.copy2(stats_src, os.path.join(args.output_dir, "global_norm_stats.json"))

    if not args.mif_path:
        print("[5/5] Syncing mIF (fallback copy)...")
        mif_files = sorted([f for f in os.listdir(mif_pkl_dir) if f.endswith(".pkl")])
        for fname in tqdm(mif_files):
            dst = os.path.join(mif_out, fname)
            if not os.path.exists(dst):
                with open(os.path.join(mif_pkl_dir, fname), "rb") as f:
                    data = pickle.load(f)
                with open(dst, "wb") as f:
                    pickle.dump(data, f)

    print("Done! New features saved to:", args.output_dir)
    print(f"  Samples: {len(he_files)} patches")

if __name__ == "__main__":
    main()
