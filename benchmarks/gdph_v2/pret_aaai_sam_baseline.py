from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from benchmarks.gdph_v2.pret_utils import (
    PRET_DIR,
    binary_segmentation_metrics,
    box_original_to_target,
    read_csv,
    write_csv_atomic,
    write_json_atomic,
)


CLASS_NAMES = [
    "tumor_epithelium",
    "tumor_stroma",
    "background",
    "necrosis",
    "normal_gland",
    "normal_stroma",
    "submucosa_serosa",
    "muscle",
    "lymphocyte_aggregate",
    "mucus",
    "fat",
    "blood",
]


def _boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int) -> float:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if not np.any(truth) and not np.any(pred):
        return 1.0
    if not np.any(truth) or not np.any(pred):
        return 0.0
    truth_boundary = truth ^ ndi.binary_erosion(truth)
    pred_boundary = pred ^ ndi.binary_erosion(pred)
    if not np.any(truth_boundary) or not np.any(pred_boundary):
        return 0.0
    truth_dist = ndi.distance_transform_edt(~truth_boundary)
    pred_dist = ndi.distance_transform_edt(~pred_boundary)
    precision = float(np.mean(truth_dist[pred_boundary] <= tolerance))
    recall = float(np.mean(pred_dist[truth_boundary] <= tolerance))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _dedupe_prompts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        seen.setdefault((row["query_id"], row["prompt_source"], int(row["shot"])), row)
    return [seen[key] for key in sorted(seen)]


def _load_predictor(checkpoint: str, model_type: str, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def _crop_box(box: tuple[int, int, int, int], width: int, height: int, margin: float, max_dim: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    crop_w = min(width, max(bw * (1 + margin * 2), bw, 1))
    crop_h = min(height, max(bh * (1 + margin * 2), bh, 1))
    scale = min(1.0, max_dim / max(crop_w, crop_h))
    crop_w *= scale
    crop_h *= scale
    left = int(max(0, min(width - 1, round(cx - crop_w / 2))))
    top = int(max(0, min(height - 1, round(cy - crop_h / 2))))
    right = int(min(width, max(left + 1, round(cx + crop_w / 2))))
    bottom = int(min(height, max(top + 1, round(cy + crop_h / 2))))
    return left, top, right, bottom


def _evaluate_image(
    source_root: str,
    prompts: list[dict[str, str]],
    checkpoint: str,
    model_type: str,
    model_name: str,
    device: str,
    crop_margin: float,
    max_crop_dim: int,
) -> tuple[str, list[dict]]:
    source = Path(source_root)
    image_id = prompts[0]["image_id"]
    superpixel_dir = source / PRET_DIR / image_id
    validation = json.loads((superpixel_dir / "validation.json").read_text(encoding="utf-8"))
    original_size = tuple(int(v) for v in validation["original_size"])
    target_shape = tuple(int(v) for v in validation["gt_mask_shape"])
    rgb = np.load(superpixel_dir / "he_10x_rgb.npy", mmap_mode="r")
    gt = np.load(source / "masks" / f"{image_id}_gt_mask.npy", mmap_mode="r")
    if tuple(gt.shape) != target_shape:
        raise RuntimeError(f"{image_id} GT shape mismatch: {gt.shape} vs {target_shape}")
    predictor = _load_predictor(checkpoint, model_type, device)
    rows: list[dict] = []
    height, width = target_shape
    for prompt in prompts:
        class_id = int(prompt["class_id"])
        original_box = (
            float(prompt["x0_original"]),
            float(prompt["y0_original"]),
            float(prompt["x1_original"]),
            float(prompt["y1_original"]),
        )
        box = box_original_to_target(original_box, original_size, target_shape)
        crop = _crop_box(box, width, height, crop_margin, max_crop_dim)
        cx0, cy0, cx1, cy1 = crop
        image_crop = np.asarray(rgb[cy0:cy1, cx0:cx1, :], dtype=np.uint8)
        gt_crop = np.asarray(gt[cy0:cy1, cx0:cx1] == class_id)
        local_box = np.asarray([box[0] - cx0, box[1] - cy0, box[2] - cx0, box[3] - cy0], dtype=np.float32)
        predictor.set_image(image_crop)
        masks, scores, _ = predictor.predict(box=local_box, multimask_output=True)
        best = int(np.argmax(scores))
        pred = np.asarray(masks[best], dtype=bool)
        binary = binary_segmentation_metrics(gt_crop, pred)
        rows.append(
            {
                "query_id": prompt["query_id"],
                "image_id": image_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                "model": model_name,
                "prompt_source": prompt["prompt_source"],
                "scope": "local_box_crop",
                "sam_score": float(scores[best]),
                "crop_width": int(cx1 - cx0),
                "crop_height": int(cy1 - cy0),
                "dice": float(binary["dice"]),
                "iou": float(binary["iou"]),
                "binary_miou": float(binary["iou"]),
                "precision": float(binary["precision"]),
                "recall": float(binary["recall"]),
                "boundary_f1_5px": _boundary_f1(gt_crop, pred, 5),
                "boundary_f1_10px": _boundary_f1(gt_crop, pred, 10),
            }
        )
    return image_id, rows


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM/MedSAM same realistic_box local prompt baseline.")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--model_name", default="sam_vit_b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--crop_margin", type=float, default=1.0)
    parser.add_argument("--max_crop_dim", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / PRET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"{args.model_name}_baseline_status.json"
    if not Path(args.checkpoint).is_file():
        write_json_atomic(status_path, {"passed": False, "status": "checkpoint_missing", "checkpoint": args.checkpoint})
        print(json.dumps({"status": "checkpoint_missing", "checkpoint": args.checkpoint}, indent=2))
        return
    try:
        import segment_anything  # noqa: F401
        import torch
    except Exception as exc:
        write_json_atomic(status_path, {"passed": False, "status": "import_failed", "error": str(exc)})
        print(json.dumps({"status": "import_failed", "error": str(exc)}, indent=2))
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    source = Path(args.source_root)
    prompts_path = Path(args.prompts_csv) if args.prompts_csv else source / PRET_DIR / "prompts.csv"
    prompts = _dedupe_prompts([row for row in read_csv(prompts_path) if row["prompt_source"] == "realistic_box"])
    if args.image_id:
        requested = set(args.image_id)
        prompts = [row for row in prompts if row["image_id"] in requested]
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in prompts:
        by_image[prompt["image_id"]].append(prompt)
    results: list[dict] = []
    # CUDA cannot be initialized safely in forked subprocesses. Keep the default
    # single-worker path in-process; use multi-process only for explicit CPU runs.
    if args.workers <= 1:
        for completed, (image_id, image_prompts) in enumerate(by_image.items(), start=1):
            _, rows = _evaluate_image(
                args.source_root,
                image_prompts,
                args.checkpoint,
                args.model_type,
                args.model_name,
                device,
                args.crop_margin,
                args.max_crop_dim,
            )
            results.extend(rows)
            print(f"{args.model_name}_baseline {completed}/{len(by_image)} image_id={image_id} rows={len(rows)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_image,
                    args.source_root,
                    image_prompts,
                    args.checkpoint,
                    args.model_type,
                    args.model_name,
                    device,
                    args.crop_margin,
                    args.max_crop_dim,
                ): image_id
                for image_id, image_prompts in by_image.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                image_id = futures[future]
                _, rows = future.result()
                results.extend(rows)
                print(f"{args.model_name}_baseline {completed}/{len(futures)} image_id={image_id} rows={len(rows)}", flush=True)
    if not results:
        write_json_atomic(status_path, {"passed": False, "status": "no_results"})
        return
    metrics_path = output_dir / f"{args.model_name}_baseline_metrics.csv"
    write_csv_atomic(metrics_path, results)
    summary = []
    for class_id in sorted({int(row["class_id"]) for row in results}):
        subset = [row for row in results if int(row["class_id"]) == class_id]
        summary.append(
            {
                "model": args.model_name,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                "queries": len(subset),
                "mean_dice": _mean(subset, "dice"),
                "mean_iou": _mean(subset, "iou"),
                "mean_boundary_f1_5px": _mean(subset, "boundary_f1_5px"),
                "mean_boundary_f1_10px": _mean(subset, "boundary_f1_10px"),
            }
        )
    summary.insert(
        0,
        {
            "model": args.model_name,
            "class_id": -1,
            "class_name": "overall",
            "queries": len(results),
            "mean_dice": _mean(results, "dice"),
            "mean_iou": _mean(results, "iou"),
            "mean_boundary_f1_5px": _mean(results, "boundary_f1_5px"),
            "mean_boundary_f1_10px": _mean(results, "boundary_f1_10px"),
        },
    )
    write_csv_atomic(output_dir / f"{args.model_name}_baseline_summary.csv", summary)
    write_json_atomic(status_path, {"passed": True, "status": "completed", "metrics": str(metrics_path), "results": len(results)})
    print(json.dumps({"metrics": str(metrics_path), "rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()
