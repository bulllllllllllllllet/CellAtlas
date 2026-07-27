#!/usr/bin/env python3
"""Freeze GT-guided WSI and patch prompts for global-vs-local evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--tile-index", type=Path, action="append", required=True)
    parser.add_argument("--class-config", type=Path, required=True)
    parser.add_argument("--target-class", required=True, help="Configured class name or integer ID")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/nfs-medical3/zyh/v4/baseline"),
    )
    parser.add_argument("--timestamp")
    return parser.parse_args()


def stable_seed(global_seed: int, *parts: object) -> int:
    payload = "\0".join([str(global_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def sample_pixel_center(mask: np.ndarray, seed: int) -> list[float] | None:
    flat = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(flat):
        return None
    chosen = int(flat[np.random.default_rng(seed).integers(len(flat))])
    y, x = np.unravel_index(chosen, mask.shape)
    return [float(x) + 0.5, float(y) + 0.5]


def load_class_contract(path: Path, target: str) -> tuple[int, tuple[int, int, int], int]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    class_map = config["data"]["class_map"]
    matches = [
        item for item in class_map
        if str(item["id"]) == target or str(item["name"]) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one class matching {target!r}, found {len(matches)}")
    item = matches[0]
    return int(item["id"]), tuple(int(value) for value in item["rgb"]), int(config["data"]["ignore_index"])


def rgb_crop_with_valid(
    image: pyvips.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    target_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((height, width), dtype=bool)
    valid = np.zeros((height, width), dtype=bool)
    crop_width = min(width, max(0, image.width - x))
    crop_height = min(height, max(0, image.height - y))
    if crop_width <= 0 or crop_height <= 0:
        return target, valid
    crop = image.crop(x, y, crop_width, crop_height).extract_band(0, n=3)
    array = np.ndarray(
        buffer=crop.write_to_memory(),
        dtype=np.uint8,
        shape=(crop_height, crop_width, 3),
    )
    valid[:crop_height, :crop_width] = True
    target[:crop_height, :crop_width] = np.all(
        array == np.asarray(target_rgb, dtype=np.uint8),
        axis=2,
    )
    return target, valid


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"wsi_gt_guided_tasks_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    target_id, target_rgb, ignore_index = load_class_contract(args.class_config, args.target_class)
    cohort = pd.read_parquet(args.cohort_manifest)
    cohort = cohort.loc[cohort["split"].eq(args.split)].set_index("wsi_id", drop=False)
    task_rows: list[dict] = []

    for tile_path in args.tile_index:
        tiles = pd.read_parquet(tile_path).sort_values("source_index").reset_index(drop=True)
        if tiles["wsi_id"].nunique() != 1:
            raise ValueError(f"tile index must contain exactly one WSI: {tile_path}")
        wsi_id = str(tiles.iloc[0]["wsi_id"])
        if wsi_id not in cohort.index:
            raise ValueError(f"{wsi_id} is absent from requested cohort split {args.split}")
        cohort_row = cohort.loc[wsi_id]
        gt_path = Path(cohort_row["gt_path"])
        gt = pyvips.Image.new_from_file(str(gt_path), access="random")
        rows: list[dict] = []

        for tile in tiles.to_dict("records"):
            width = int(tile["width_10x"])
            height = int(tile["height_10x"])
            target, valid = rgb_crop_with_valid(
                gt,
                int(tile["x_10x"]),
                int(tile["y_10x"]),
                width,
                height,
                target_rgb,
            )
            positive = sample_pixel_center(
                target & valid,
                stable_seed(args.seed, wsi_id, target_id, tile["patch_id"], "positive"),
            )
            negative = sample_pixel_center(
                (~target) & valid,
                stable_seed(args.seed, wsi_id, target_id, tile["patch_id"], "negative"),
            ) if positive is not None else None
            row = dict(tile)
            row.update({
                "gt_path": str(gt_path),
                "target_class": target_id,
                "target_class_name": str(args.target_class),
                "valid_pixels": int(valid.sum()),
                "target_pixels": int((target & valid).sum()),
                "has_target": positive is not None,
                "positive_point_10x": json.dumps(positive, separators=(",", ":")) if positive else None,
                "negative_point_10x": json.dumps(negative, separators=(",", ":")) if negative else None,
                "negative_available": negative is not None,
                "positive_audit": bool(
                    positive is not None
                    and target[int(np.floor(positive[1])), int(np.floor(positive[0]))]
                ),
                "negative_audit": bool(
                    negative is None
                    or not target[int(np.floor(negative[1])), int(np.floor(negative[0]))]
                ),
            })
            rows.append(row)

        patch_frame = pd.DataFrame(rows)
        target_rows = patch_frame.loc[patch_frame["has_target"]].reset_index(drop=True)
        if target_rows.empty:
            raise RuntimeError(f"no target pixels for {wsi_id}, class={target_id}")
        seed_index = int(
            np.random.default_rng(stable_seed(args.seed, wsi_id, target_id, "wsi_seed"))
            .integers(len(target_rows))
        )
        seed_row = target_rows.iloc[seed_index]
        positive_local = json.loads(seed_row["positive_point_10x"])
        negative_local = (
            json.loads(seed_row["negative_point_10x"])
            if seed_row["negative_available"]
            else None
        )
        downsample = float(seed_row["level0_downsample"])

        def to_level0(point: list[float]) -> dict[str, float]:
            return {
                "x": float(seed_row["x_level0"]) + point[0] * downsample,
                "y": float(seed_row["y_level0"]) + point[1] * downsample,
            }

        prompt = {
            "coordinate_space": "level0",
            "prompt_size": "point",
            "positive": [to_level0(positive_local)],
            "negative": [to_level0(negative_local)] if negative_local else [],
        }
        patch_path = output / f"patch_prompts_{wsi_id}_{stamp}.parquet"
        prompt_path = output / f"j5_prompt_{wsi_id}_{stamp}.json"
        patch_frame.to_parquet(patch_path, index=False)
        prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
        task_rows.append({
            "wsi_id": wsi_id,
            "target_class": target_id,
            "target_class_name": str(args.target_class),
            "tile_index": str(tile_path),
            "tile_index_sha256": sha256_path(tile_path),
            "gt_path": str(gt_path),
            "wsi_path": str(seed_row["wsi_path"]),
            "patch_prompt_manifest": str(patch_path),
            "patch_prompt_manifest_sha256": sha256_path(patch_path),
            "j5_prompt_json": str(prompt_path),
            "j5_prompt_json_sha256": sha256_path(prompt_path),
            "seed_patch_id": str(seed_row["patch_id"]),
            "n_patch": int(len(patch_frame)),
            "n_target": int(patch_frame["has_target"].sum()),
            "positive_points": int(patch_frame["has_target"].sum()),
            "negative_points": int(patch_frame["negative_available"].sum()),
            "sam_calls_per_method": int(patch_frame["has_target"].sum()),
        })

    task_path = output / f"wsi_tasks_{stamp}.parquet"
    pd.DataFrame(task_rows).to_parquet(task_path, index=False)
    metadata = {
        "timestamp": stamp,
        "split": args.split,
        "target_class": target_id,
        "target_rgb": list(target_rgb),
        "ignore_index": ignore_index,
        "seed": args.seed,
        "wsi_tasks": str(task_path),
        "wsi_tasks_sha256": sha256_path(task_path),
        "wsi_count": len(task_rows),
        "protocol": "GT-guided patch localization and random positive/negative pixel centers",
        "j5_seed_protocol": "one frozen GT-simulated positive/negative query per WSI",
        "oracle_disclosure_required": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "tasks": len(task_rows)}, indent=2))


if __name__ == "__main__":
    main()
