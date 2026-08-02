#!/usr/bin/env python3
"""Evaluate prompted and held-out TLS polygons against a stitched WSI prediction."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pyvips

from benchmarks.v4.test_TLS.prepare_tls_case import load_polygons
from module.KFBreader.kfbreader import KFBSlide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--timestamp")
    return parser.parse_args()


def read_mask(path: Path, height: int, width: int) -> np.ndarray:
    image = pyvips.Image.new_from_file(str(path), access="sequential").extract_band(0)
    if (image.height, image.width) != (height, width):
        raise ValueError(f"mask shape {(image.height, image.width)} != {(height, width)}")
    return np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8, shape=(height, width))


def counts(prediction: np.ndarray, target: np.ndarray) -> dict:
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def optional_counts(prediction: np.ndarray, target: np.ndarray) -> dict | None:
    return counts(prediction, target) if bool(target.any()) else None


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32).copy()
    active = mask.astype(bool)
    result[active] = result[active] * 0.55 + np.asarray(color, np.float32) * 0.45
    return np.clip(result, 0, 255).astype(np.uint8)


def mark_prompts(
    image: np.ndarray, prompts: dict, scale_x: float, scale_y: float,
) -> np.ndarray:
    result = image.copy()
    for sign, color in (("positive", (255, 255, 0)), ("negative", (0, 255, 255))):
        for point in prompts[sign]:
            center = (int(round(float(point["x"]) * scale_x)), int(round(float(point["y"]) * scale_y)))
            cv2.circle(result, center, 8, color, 3, lineType=cv2.LINE_AA)
            cv2.line(result, (center[0] - 11, center[1]), (center[0] + 11, center[1]), color, 2)
            cv2.line(result, (center[0], center[1] - 11), (center[0], center[1] + 11), color, 2)
    return result


def labelled_panel(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (max(320, 18 * len(label)), 42), (0, 0, 0), -1)
    cv2.putText(result, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return result


def write_review_panel(
    path: Path, wsi_path: Path, prediction: np.ndarray, target: np.ndarray, prompts: dict,
) -> dict:
    slide = KFBSlide(str(wsi_path))
    level = slide.level_count - 1
    thumb_width, thumb_height = map(int, slide.level_dimensions[level])
    thumbnail = np.asarray(
        slide.read_region((0, 0), level, (thumb_width, thumb_height)), dtype=np.uint8,
    )[..., :3]
    gt_small = cv2.resize(target.astype(np.uint8), (thumb_width, thumb_height), interpolation=cv2.INTER_NEAREST)
    pred_small = cv2.resize(
        prediction.astype(np.uint8), (thumb_width, thumb_height), interpolation=cv2.INTER_NEAREST,
    )
    scale_x = thumb_width / float(slide.dimensions[0])
    scale_y = thumb_height / float(slide.dimensions[1])
    gt_overlay = mark_prompts(overlay(thumbnail, gt_small, (0, 255, 0)), prompts, scale_x, scale_y)
    pred_overlay = mark_prompts(overlay(thumbnail, pred_small, (255, 0, 0)), prompts, scale_x, scale_y)
    gt_binary = mark_prompts(np.repeat((gt_small * 255)[..., None], 3, axis=2), prompts, scale_x, scale_y)
    pred_binary = mark_prompts(np.repeat((pred_small * 255)[..., None], 3, axis=2), prompts, scale_x, scale_y)
    panel = np.concatenate([
        np.concatenate([
            labelled_panel(gt_overlay, "HE + TLS GT (green)"),
            labelled_panel(pred_overlay, "HE + prediction (red)"),
        ], axis=1),
        np.concatenate([
            labelled_panel(gt_binary, "TLS GT binary"),
            labelled_panel(pred_binary, "Prediction binary"),
        ], axis=1),
    ], axis=0)
    if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write review panel: {path}")
    return {
        "path": str(path), "thumbnail_shape": [thumb_height, thumb_width],
        "panel_shape": list(panel.shape[:2]), "visual_review": "pending_user_review",
    }


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = json.loads((args.inference_dir / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError(f"inference is not complete: {metadata.get('status')}")
    case = json.loads(args.case_manifest.read_text())
    height, width = map(int, metadata["output_shape_10x"])
    prediction = read_mask(Path(metadata["mask_tiff"]), height, width) > 0
    polygons = load_polygons(args.annotation_json)
    downsample = int(metadata["level0_downsample"])
    wsi_path = Path(case["wsi_path"])
    if Path(metadata["wsi_path"]) != wsi_path:
        raise ValueError(f"source identity mismatch: {metadata['wsi_path']} != {wsi_path}")
    if [height, width] != list(map(int, case["output_shape_10x"])):
        raise ValueError("inference/case 10x output shape mismatch")
    prompts = json.loads(Path(case["prompt_json"]).read_text(encoding="utf-8"))
    per_polygon = []
    union = np.zeros((height, width), np.uint8)
    prompted_union = np.zeros_like(union)
    held_out_union = np.zeros_like(union)
    prompted = set(map(int, case["positive_polygon_indices"]))
    for index, polygon in enumerate(polygons):
        target = np.zeros_like(union)
        cv2.fillPoly(target, [np.rint(polygon / downsample).astype(np.int32)], 1)
        union |= target
        if index in prompted:
            prompted_union |= target
        else:
            held_out_union |= target
        per_polygon.append({
            "polygon_index": index, "prompted": index in prompted,
            "area_pixels_10x": int(target.sum()), **counts(prediction, target.astype(bool)),
        })
    review = write_review_panel(
        args.inference_dir / f"tls_review_panel_{stamp}.png",
        wsi_path, prediction, union.astype(bool), prompts,
    )
    prompted_rows = [row for row in per_polygon if row["prompted"]]
    held_out_rows = [row for row in per_polygon if not row["prompted"]]
    held_out_target = held_out_union.astype(bool)
    held_out_evaluation_domain = ~prompted_union.astype(bool)
    held_out_prediction = prediction & held_out_evaluation_domain
    report = {
        "timestamp": stamp, "inference_dir": str(args.inference_dir),
        "case_manifest": str(args.case_manifest), "annotation_json": str(args.annotation_json),
        "union": counts(prediction, union.astype(bool)),
        "prompted_union": optional_counts(prediction, prompted_union.astype(bool)),
        "held_out_union": optional_counts(prediction, held_out_union.astype(bool)),
        "held_out_censored_prompted_region": optional_counts(
            held_out_prediction, held_out_target
        ),
        "per_polygon": per_polygon,
        "prompted_polygon_count": len(prompted_rows),
        "held_out_polygon_count": len(held_out_rows),
        "prompted_macro_recall": float(np.mean([row["recall"] for row in prompted_rows])),
        "held_out_macro_recall": (
            float(np.mean([row["recall"] for row in held_out_rows])) if held_out_rows else None
        ),
        "held_out_retrieved_recall_ge_0_1": (
            float(np.mean([row["recall"] >= 0.1 for row in held_out_rows])) if held_out_rows else None
        ),
        "held_out_retrieved_recall_ge_0_5": (
            float(np.mean([row["recall"] >= 0.5 for row in held_out_rows])) if held_out_rows else None
        ),
        "review_panel": review,
        "binary_contract": {"prediction_values": [False, True], "ground_truth_values": [False, True]},
        "interpretation": (
            "held-out recall measures prompt-driven retrieval; held_out_censored_prompted_region "
            "excludes prompted TLS pixels from the evaluation domain; human visual review remains pending"
        ),
    }
    output = args.inference_dir / f"tls_retrieval_report_{stamp}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
