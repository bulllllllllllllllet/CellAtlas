import os, sys, torch, numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.dirname(__file__))
from XCellFormer import XCellFormer

# 1. Load new model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XCellFormer(
    input_dim=768, hidden_dim=512, n_heads=8, num_layers=4,
    output_dim=64, max_cells=255, use_large_vit=False,
).to(device)
best_ckpt = "experiments/20260510_220559/he_model_best.pth"
ckpt = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()
print(f"[OK] New model loaded from {best_ckpt}")

# 2. CTransPath (using centered crop + soft attention like training)
from utils import load_cellpose_model
from module.TransPath.ctran import ctranspath
cellpose_model = load_cellpose_model(device=device)
ctp = ctranspath().to(device)
ctp.eval()
ctp.head = torch.nn.Identity()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Test image (same as before)
img_path = "/nfs-medical3/zyh/XCellAligner_feature_extract_output/CRC02/he_images/he_x0_y27648.png"
img_np = np.array(Image.open(img_path).convert("RGB"))
h, w = img_np.shape[:2]

# Cellpose segmentation
masks, _, _ = cellpose_model.eval(img_np, diameter=18, channels=[0, 0])
unique_labels = np.unique(masks)
unique_labels = unique_labels[unique_labels != 0]
print(f"\nImage: {img_np.shape}, Cells detected: {len(unique_labels)}")

# 4. Extract features using centered crop + soft attention
centers = ndimage.center_of_mass(np.ones_like(masks), labels=masks, index=unique_labels)
cells = sorted(zip(unique_labels, centers), key=lambda x: (x[1][0], x[1][1]))
cells = cells[:255]

CROP_SZ = 128
HALF = CROP_SZ // 2

def crop_center(img, cy, cx):
    y0 = max(0, cy - HALF)
    y1 = min(h, cy + HALF)
    x0 = max(0, cx - HALF)
    x1 = min(w, cx + HALF)
    patch = img[y0:y1, x0:x1]
    pt = max(0, HALF - cy)
    pb = max(0, cy + HALF - h)
    pl = max(0, HALF - cx)
    pr = max(0, cx + HALF - w)
    if pt or pb or pl or pr:
        patch = np.pad(patch, ((pt, pb), (pl, pr), (0, 0)), mode='edge')
    return patch.astype(np.float32) / 255.0

crops = []
for label, (cy, cx) in cells:
    cy_i, cx_i = int(cy), int(cx)
    crop = crop_center(img_np, cy_i, cx_i)

    # soft attention
    label_mask = (masks[max(0,cy_i-HALF):min(h,cy_i+HALF), max(0,cx_i-HALF):min(w,cx_i+HALF)] == label).astype(np.float32)
    pt = max(0, HALF - cy_i)
    pb = max(0, cy_i + HALF - h)
    pl = max(0, HALF - cx_i)
    pr = max(0, cx_i + HALF - w)
    if pt or pb or pl or pr:
        label_mask = np.pad(label_mask, ((pt, pb), (pl, pr)), mode='edge')
    if label_mask.sum() > 0:
        blurred = ndimage.gaussian_filter(label_mask, sigma=12, mode='nearest')
        attn = 0.35 + 0.65 * (blurred / (blurred.max() + 1e-6))
        crop = crop * attn[..., None]

    crop_pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    crops.append(preprocess(crop_pil))

# Batch CTransPath
features = []
for i in range(0, len(crops), 64):
    batch = torch.stack(crops[i:i+64]).to(device)
    with torch.no_grad():
        out = ctp(batch)
    if out.dim() == 4 and out.shape[-1] == 768:
        out = out.permute(0, 3, 1, 2)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)
    features.append(out.cpu().numpy())
features = np.concatenate(features, axis=0)
n = features.shape[0]
print(f"Extracted {n} cell features")

# 5. Model inference
padded = np.zeros((255, 768), dtype=np.float32)
padded[:n] = features
mask_arr = np.zeros(255, dtype=np.float32)
mask_arr[:n] = 1.0

x_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
m_tensor = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    _, _, proj_out, _ = model(raw_images=None, x=x_tensor, mask=m_tensor)

valid_out = proj_out[0, :n].cpu().numpy()

# Check diversity
mean_vec = valid_out.mean(axis=0, keepdims=True)
var_per_dim = ((valid_out - mean_vec) ** 2).mean()
cos_sim = (valid_out @ valid_out.T)
cos_sim = cos_sim / (np.linalg.norm(valid_out, axis=1, keepdims=True) * np.linalg.norm(valid_out, axis=1))
avg_cos = (cos_sim.sum() - n) / (n * (n - 1))
print(f"Output variance: {var_per_dim:.6f}, Avg pairwise cosine: {avg_cos:.4f}")

# 6. KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(valid_out)
dist_str = ", ".join(f"{i}:{int((cluster_labels==i).sum())}" for i in range(k))
print(f"Cluster distribution: {dist_str}")
for i in range(n):
    print(f"  Cell {i}: cluster={cluster_labels[i]}, first5d={valid_out[i][:5].round(3)}")

# 7. Visualize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
colors = plt.cm.get_cmap("tab10", k)
result = img_np.copy()
for i, (label, (cy, cx)) in enumerate(cells):
    cid = cluster_labels[i]
    mask_cell = masks == label
    color = (np.array(colors(cid)[:3]) * 255).astype(np.uint8)
    for c in range(3):
        result[mask_cell, c] = result[mask_cell, c] * 0.5 + color[c] * 0.5

save_path = "/home/zyh/NewMedLabel/XCellAligner/test_cluster_result.png"
Image.fromarray(result).save(save_path)
print(f"\nResult saved to: {save_path}")
