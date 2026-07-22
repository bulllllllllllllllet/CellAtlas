#!/usr/bin/env python3
"""Export one representative multi-scale patch with parent-child overlays."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
# libvips (via read_he_patch) before torchvision (via DeepRegionEncoder).
from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch  # noqa: F401
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import (
    export_patch_multiscale,
    load_cell_module,
    load_phase2,
)
from benchmarks.v4.phase_4_cross_scale.src.geometry import edge_invariants


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase3-config", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--patch-id", type=str)
    parser.add_argument("--split", default="val")
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/phase4/data"))
    parser.add_argument("--timestamp")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def pick_representative(df: pd.DataFrame) -> pd.Series:
    # Prefer boundary / rare with high tissue and multiple classes.
    scored = df.copy()
    scored["class_count"] = scored["present_classes"].map(lambda x: len(json.loads(x)) if isinstance(x, str) else 0)
    scored["score"] = (
        scored["boundary_fraction"].astype(float) * 2.0
        + scored["tissue_fraction"].astype(float)
        + scored["class_count"] * 0.1
        + scored["sampling_group"].isin(["boundary", "rare_class", "hard_negative"]).astype(float)
    )
    return scored.sort_values("score", ascending=False).iloc[0]


def hard_assignment_rgb(assignment: np.ndarray, palette: np.ndarray) -> np.ndarray:
    hard = assignment.argmax(0)
    return palette[hard % len(palette)]


def save_overlay(path: Path, image: np.ndarray, assignment: np.ndarray, centroids: dict, title: str):
    palette = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    region_rgb = hard_assignment_rgb(assignment, palette)
    blend = (0.55 * image.astype(np.float32) / 255.0 + 0.45 * region_rgb).clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"{title} HE")
    axes[1].imshow(blend)
    axes[1].set_title(f"{title} assignment")
    axes[2].imshow(image)
    active = centroids["active"]
    # centroids are level-0; convert to patch pixel using box stored outside — caller remaps
    axes[2].scatter(centroids["px"][active], centroids["py"][active], s=12, c="cyan", linewidths=0)
    axes[2].set_title(f"{title} centroids")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def remap_centroids_to_patch(geom: dict, box: tuple[int, int, int, int], patch_size: int = 512) -> dict:
    x0, y0, w0, h0 = box
    px = (geom["centroid_x"] - x0) / max(w0, 1) * patch_size
    py = (geom["centroid_y"] - y0) / max(h0, 1) * patch_size
    return {"px": px, "py": py, "active": geom["active"]}


def save_edge_overlay(
    path: Path,
    child_image: np.ndarray,
    child_geom: dict,
    child_box: tuple[int, int, int, int],
    parent_geom: dict,
    parent_box: tuple[int, int, int, int],
    edges: dict,
    title: str,
    patch_size: int = 512,
):
    child = remap_centroids_to_patch(child_geom, child_box, patch_size)
    # project parent centroids into child patch pixel space
    px0, py0, pw0, ph0 = parent_box
    cx0, cy0, cw0, ch0 = child_box
    parent_x = (parent_geom["centroid_x"] - cx0) / max(cw0, 1) * patch_size
    parent_y = (parent_geom["centroid_y"] - cy0) / max(ch0, 1) * patch_size
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(child_image)
    active = child_geom["active"]
    ax.scatter(child["px"][active], child["py"][active], s=18, c="lime", label="child", zorder=3)
    parent_active = parent_geom["active"]
    ax.scatter(parent_x[parent_active], parent_y[parent_active], s=28, c="red", marker="x", label="parent", zorder=3)
    for child_id in np.where(active)[0]:
        for slot in range(edges["edge_index"].shape[1]):
            parent_id = int(edges["edge_index"][child_id, slot])
            weight = float(edges["edge_weight"][child_id, slot])
            if parent_id < 0 or weight <= 0:
                continue
            ax.plot(
                [child["px"][child_id], parent_x[parent_id]],
                [child["py"][child_id], parent_y[parent_id]],
                color="yellow",
                alpha=min(1.0, 0.2 + weight),
                linewidth=0.8 + 2.0 * weight,
            )
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    args = parse()
    p2cfg = load_config(args.phase2_config)
    p3cfg = load_config(args.phase3_config)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"singleton_alignment_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    df = pd.read_parquet(args.patch_index)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.patch_id:
        rows = df[df["patch_id"] == args.patch_id]
        if rows.empty:
            raise ValueError(f"patch_id not found in split={args.split}: {args.patch_id}")
        row = rows.iloc[0]
    else:
        row = pick_representative(df)
    row_dict = row.to_dict()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if p3cfg["training"]["amp_dtype"] == "bfloat16" else torch.float16
    phase2 = load_phase2(p2cfg, args.phase2_checkpoint, device)
    cell = None
    if args.cell_checkpoint is not None:
        cell = load_cell_module(
            int(p3cfg["model"]["region_dim"]),
            int(p3cfg["data"]["cell_feature_dim"]),
            args.cell_checkpoint,
            device,
        )
        # empty cells still exercise the fusion path with zero density
        max_cells = int(p3cfg["data"]["max_cells_per_patch"])
        cells = torch.zeros((1, max_cells, int(p3cfg["data"]["cell_feature_dim"])), dtype=torch.float32)
        cell_valid = torch.zeros((1, max_cells), dtype=torch.bool)
        total = torch.zeros((1,), dtype=torch.float32)
    else:
        cells = cell_valid = total = None

    result = export_patch_multiscale(
        row_dict,
        phase2,
        cell,
        device,
        amp,
        cells=cells,
        cell_valid=cell_valid,
        total_cell_count=total,
        top_k=args.top_k,
    )

    # save tensors / arrays
    np.savez_compressed(
        output / "tokens.npz",
        fine_tokens=result["scales"]["10x"]["tokens"],
        middle_tokens=result["scales"]["5x"]["tokens"],
        coarse_tokens=result["scales"]["2p5x"]["tokens"],
        fine_region_tokens=result["scales"]["10x"]["region_tokens"],
        fine_middle_edge_index=result["edges"]["fine_middle"]["edge_index"],
        fine_middle_edge_weight=result["edges"]["fine_middle"]["edge_weight"],
        middle_coarse_edge_index=result["edges"]["middle_coarse"]["edge_index"],
        middle_coarse_edge_weight=result["edges"]["middle_coarse"]["edge_weight"],
        fine_centroid_x=result["scales"]["10x"]["centroid_x"],
        fine_centroid_y=result["scales"]["10x"]["centroid_y"],
        middle_centroid_x=result["scales"]["5x"]["centroid_x"],
        middle_centroid_y=result["scales"]["5x"]["centroid_y"],
        coarse_centroid_x=result["scales"]["2p5x"]["centroid_x"],
        coarse_centroid_y=result["scales"]["2p5x"]["centroid_y"],
        fine_active=result["scales"]["10x"]["active"],
        middle_active=result["scales"]["5x"]["active"],
        coarse_active=result["scales"]["2p5x"]["active"],
    )
    for scale in ("10x", "5x", "2p5x"):
        np.save(output / f"assignment_{scale}.npy", result["scales"][scale]["assignment"].astype(np.float16))
        plt.imsave(output / f"he_{scale}.png", result["scales"][scale]["image_rgb"])
        geom_plot = remap_centroids_to_patch(result["scales"][scale], result["scales"][scale]["box_level0"])
        save_overlay(
            output / f"overlay_{scale}.png",
            result["scales"][scale]["image_rgb"],
            result["scales"][scale]["assignment"],
            geom_plot,
            scale,
        )

    save_edge_overlay(
        output / "edges_fine_middle.png",
        result["scales"]["10x"]["image_rgb"],
        result["scales"]["10x"],
        result["scales"]["10x"]["box_level0"],
        result["scales"]["5x"],
        result["scales"]["5x"]["box_level0"],
        result["edges"]["fine_middle"],
        "fine→middle edges",
    )
    save_edge_overlay(
        output / "edges_middle_coarse.png",
        result["scales"]["5x"]["image_rgb"],
        result["scales"]["5x"],
        result["scales"]["5x"]["box_level0"],
        result["scales"]["2p5x"],
        result["scales"]["2p5x"]["box_level0"],
        result["edges"]["middle_coarse"],
        "middle→coarse edges",
    )

    inv_fm = edge_invariants(result["edges"]["fine_middle"])
    inv_mc = edge_invariants(result["edges"]["middle_coarse"])
    # physical nesting checks
    fbox = result["scales"]["10x"]["box_level0"]
    mbox = result["scales"]["5x"]["box_level0"]
    cbox = result["scales"]["2p5x"]["box_level0"]
    nesting = {
        "fine_inside_middle": bool(
            fbox[0] >= mbox[0]
            and fbox[1] >= mbox[1]
            and fbox[0] + fbox[2] <= mbox[0] + mbox[2]
            and fbox[1] + fbox[3] <= mbox[1] + mbox[3]
        ),
        "middle_inside_coarse": bool(
            mbox[0] >= cbox[0]
            and mbox[1] >= cbox[1]
            and mbox[0] + mbox[2] <= cbox[0] + cbox[2]
            and mbox[1] + mbox[3] <= cbox[1] + cbox[3]
        ),
        "fine_box": fbox,
        "middle_box": mbox,
        "coarse_box": cbox,
    }
    report = {
        "patch_id": result["patch_id"],
        "wsi_id": result["wsi_id"],
        "split": args.split,
        "top_k": args.top_k,
        "phase2_checkpoint": str(args.phase2_checkpoint),
        "cell_checkpoint": str(args.cell_checkpoint) if args.cell_checkpoint else None,
        "nesting": nesting,
        "fine_middle_invariants": inv_fm,
        "middle_coarse_invariants": inv_mc,
        "passed": bool(inv_fm["passed"] and inv_mc["passed"] and nesting["fine_inside_middle"] and nesting["middle_inside_coarse"]),
        "active_counts": {
            "10x": int(result["scales"]["10x"]["active"].sum()),
            "5x": int(result["scales"]["5x"]["active"].sum()),
            "2p5x": int(result["scales"]["2p5x"]["active"].sum()),
        },
    }
    (output / "invariant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({"event": "singleton_complete", "output": str(output), **report}, flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
