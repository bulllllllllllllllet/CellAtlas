#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmarks.v4.phase_1_multiscale.src.common import create_run_dir, load_config, save_json
from benchmarks.v4.phase_1_multiscale.src.data import read_pairs, rgb_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit explicit GDPH HE/GT pairs without modifying source data.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--num-previews", type=int, default=20)
    p.add_argument("--limit", type=int, help="Deterministic sample size; omit for every pair.")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--timestamp")
    return p.parse_args()


def preview(pair, output: Path) -> None:
    import pyvips
    source = pyvips.Image.new_from_file(str(pair.he_path), access="sequential")
    he_thumb = source.thumbnail_image(768)
    he_array = np.ndarray(buffer=he_thumb.write_to_memory(), dtype=np.uint8, shape=(he_thumb.height, he_thumb.width, he_thumb.bands))
    he = Image.fromarray(he_array[..., :3]).convert("RGB")
    gt_source = pyvips.Image.new_from_file(str(pair.gt_path), access="sequential")
    gt_thumb = gt_source.resize(he.width / gt_source.width, vscale=he.height / gt_source.height, kernel="nearest")
    gt_array = np.ndarray(buffer=gt_thumb.write_to_memory(), dtype=np.uint8, shape=(gt_thumb.height, gt_thumb.width, gt_thumb.bands))
    gt = Image.fromarray(gt_array[..., :3]).convert("RGB")
    overlay = Image.blend(he, gt, 0.45)
    sheet = Image.new("RGB", (he.width * 3, he.height + 28), "white")
    for i, panel in enumerate((he, gt, overlay)): sheet.paste(panel, (i * he.width, 28))
    draw = ImageDraw.Draw(sheet)
    for i, title in enumerate(("HE", "GT", "HE + GT")): draw.text((i * he.width + 5, 6), title, fill="black")
    sheet.save(output / f"{pair.wsi_id}_overlay.png")


def count_pair(pair):
    absent = [str(p) for p in (pair.he_path, pair.gt_path, pair.nuclei_class_path, pair.nuclei_instance_path) if not p.is_file()]
    if absent:
        return pair.wsi_id, absent, None
    return pair.wsi_id, [], rgb_counts(pair.gt_path)


def main() -> None:
    args, cfg = parse_args(), load_config(parse_args().config)
    out = create_run_dir(cfg, "audit", args.timestamp)
    pairs, missing, duplicate_paths, color_totals = read_pairs(cfg), [], Counter(), Counter()
    if args.limit:
        if not 1 <= args.limit <= len(pairs): raise ValueError(f"--limit must be in 1..{len(pairs)}")
        rng = np.random.default_rng(args.seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=args.limit, replace=False))]
    if args.num_workers < 1: raise ValueError("--num-workers must be positive")
    seen_gt: set[Path] = set()
    for pair in pairs:
        if pair.gt_path in seen_gt: duplicate_paths[pair.gt_path] += 1
        seen_gt.add(pair.gt_path)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(count_pair, pair) for pair in pairs]
        for future in as_completed(futures):
            wsi_id, absent, counts = future.result()
            if absent: missing.append({"wsi_id": wsi_id, "missing": absent})
            else: color_totals.update(counts)
    preview_dir = out / "alignment_previews"; preview_dir.mkdir()
    for pair in [p for p in pairs if p.he_path.is_file() and p.gt_path.is_file()][:args.num_previews]: preview(pair, preview_dir)
    save_json(out / "gt_color_counts.json", [{"rgb": list(k), "pixels": v} for k, v in color_totals.most_common()])
    save_json(out / "audit_report.json", {"pair_count": len(pairs), "missing": missing, "duplicate_gt_paths": {str(k): v + 1 for k, v in duplicate_paths.items()}, "patient_id_policy": cfg["split"]["patient_id_policy"], "preview_count": args.num_previews, "num_workers": args.num_workers, "seed": args.seed})
    print(out)


if __name__ == "__main__": main()
