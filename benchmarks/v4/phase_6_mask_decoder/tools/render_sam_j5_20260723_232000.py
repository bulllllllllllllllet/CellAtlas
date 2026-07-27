import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = "/nfs-medical3/zyh/v4/phase6/evaluation/joint_pixel_audit_20260723_232000"
sam = np.load("/nfs-medical3/zyh/v4/baseline/sam_dependency_smoke_20260723_225700/prediction_arrays_20260723_225700.npz")
j5 = np.load(f"{root}/prediction_arrays_019522.npz")
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
items = [("HE + frozen prompts", j5["image"]), ("GT muscle", j5["target_mask"]), ("SAM prediction", sam["binary_mask"]), ("HE + frozen prompts", j5["image"]), ("GT muscle", j5["target_mask"]), ("J5 prediction", j5["binary_mask"])]
for axis, (title, image) in zip(axes.flat, items):
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None); axis.set_title(title); axis.axis("off")
for axis in (axes[0, 0], axes[1, 0]):
    axis.scatter(j5["positive_points"][:, 0], j5["positive_points"][:, 1], s=55, facecolors="none", edgecolors="lime")
    axis.scatter(j5["negative_points"][:, 0], j5["negative_points"][:, 1], s=45, marker="x", c="red")
for axis in (axes[0, 2], axes[1, 2]): axis.contour(j5["target_mask"], levels=[.5], colors="cyan", linewidths=.7)
fig.savefig(f"{root}/sam_vs_j5_20260723_232000.png", dpi=150)
