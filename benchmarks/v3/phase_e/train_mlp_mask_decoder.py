# 作用：用冻结的 Phase C small-scale retrieval score 训练 GPU MLP decoder，并以原始 GT 像素做五折 WSI-level 评估。

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.v3.phase_c.run_multiscale_baseline import (
    MULTISCALE_ROOT, NUM_CLASSES, SCORE_DIR, compute_gt_counts, load_gt_mask, read_csv,
)
from benchmarks.gdph_v2.pret_utils import stable_seed

V3_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
PRET_ROOT = V3_ROOT / "pret_superpixel"
MANIFEST = PRET_ROOT / "data_manifest_v3.csv"
METRICS = PRET_ROOT / "evaluations" / "multiscale_baseline_metrics.csv"
OUT = PRET_ROOT / "mask_decoder"
PHASE_DIR = Path(__file__).resolve().parent
SCALE = "small"
TOKEN_FILE = "tokens_image_cell_reg_texture_cellstats.npy"
SAMPLE_CACHE = OUT / "mlp_decoder_train_samples.npz"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


@dataclass
class ImageData:
    tokens: np.ndarray
    areas: np.ndarray
    centers: np.ndarray
    density: np.ndarray
    count: np.ndarray
    valid_pixels: np.ndarray
    counts: np.ndarray
    edges: np.ndarray


class Decoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.10),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_image(image_id: str, gt_path: str, cache: dict[str, ImageData]) -> ImageData:
    if image_id in cache:
        return cache[image_id]
    directory = MULTISCALE_ROOT / image_id / SCALE
    rows = read_csv(directory / "superpixels.csv")
    segments = np.load(directory / "superpixels.npy", mmap_mode="r")
    gt = load_gt_mask(Path(gt_path))
    if gt.shape != segments.shape:
        gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((segments.shape[1], segments.shape[0]), Image.Resampling.NEAREST), dtype=np.int16)
    counts, valid_pixels = compute_gt_counts(segments, gt, int(segments.max()) + 1)
    edges = np.asarray(np.load(directory / "adjacency.npy", mmap_mode="r"), dtype=np.int64)
    edges = edges[(edges[:, 0] >= 0) & (edges[:, 0] < len(rows)) & (edges[:, 1] >= 0) & (edges[:, 1] < len(rows))]
    data = ImageData(
        tokens=np.asarray(np.load(directory / TOKEN_FILE, mmap_mode="r"), dtype=np.float32),
        areas=np.asarray([float(row["area"]) for row in rows], dtype=np.float32),
        centers=np.asarray([[float(row["center_x"]), float(row["center_y"])] for row in rows], dtype=np.float32),
        density=np.asarray([float(row.get("cell_density", 0.0)) for row in rows], dtype=np.float32),
        count=np.asarray([float(row.get("cell_count", 0.0)) for row in rows], dtype=np.float32),
        valid_pixels=valid_pixels.astype(np.float64), counts=counts.astype(np.float64),
        edges=edges,
    )
    cache[image_id] = data
    return data


def node_features(data: ImageData, score_path: Path, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(score_path) as score:
        pos = score["score_pos"].astype(np.float32)
        neg = score["score_neg"].astype(np.float32)
        final = score["score_final"].astype(np.float32)
        rank = score["rank_percentile"].astype(np.float32)
        positive = score["positive_prompt_segments"].astype(np.int64)
        negative = score["negative_prompt_segments"].astype(np.int64)
    n = len(final)
    neighbor_sum = np.zeros(n, dtype=np.float32)
    neighbor_count = np.zeros(n, dtype=np.float32)
    neighbor_max = np.full(n, -np.inf, dtype=np.float32)
    if len(data.edges):
        left, right = data.edges[:, 0], data.edges[:, 1]
        np.add.at(neighbor_sum, left, final[right])
        np.add.at(neighbor_sum, right, final[left])
        np.add.at(neighbor_count, left, 1.0)
        np.add.at(neighbor_count, right, 1.0)
        np.maximum.at(neighbor_max, left, final[right])
        np.maximum.at(neighbor_max, right, final[left])
    neighbor_mean = neighbor_sum / np.maximum(neighbor_count, 1.0)
    neighbor_max[~np.isfinite(neighbor_max)] = final[~np.isfinite(neighbor_max)]
    span = np.maximum(data.centers.max(axis=0) - data.centers.min(axis=0), 1.0)
    center_norm = (data.centers - data.centers.min(axis=0)) / span

    def prompt_distance(ids: np.ndarray) -> np.ndarray:
        ids = ids[(ids >= 0) & (ids < n)]
        if not len(ids):
            return np.ones(n, dtype=np.float32)
        delta = data.centers[:, None, :] - data.centers[ids][None, :, :]
        return np.sqrt(np.sum((delta / span) ** 2, axis=2)).min(axis=1).astype(np.float32)

    extra = np.column_stack((
        pos, neg, final, rank, np.log1p(data.areas), center_norm,
        np.log1p(data.density), np.log1p(data.count), prompt_distance(positive), prompt_distance(negative),
        neighbor_mean, neighbor_max,
    )).astype(np.float32)
    selected = slice(None) if ids is None else ids
    return np.concatenate((data.tokens[selected], extra[selected]), axis=1), (data.valid_pixels > 0)[selected]


def sampled_ids(target: np.ndarray, valid: np.ndarray, seed: int, pos_limit: int, neg_limit: int, mixed_limit: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = ((target >= 0.5) & valid, (target <= 0.0) & valid, (target > 0.0) & (target < 0.5) & valid)
    limits = (pos_limit, neg_limit, mixed_limit)
    chosen = []
    for group, limit in zip(groups, limits, strict=True):
        ids = np.flatnonzero(group)
        if len(ids) > limit:
            ids = rng.choice(ids, size=limit, replace=False)
        chosen.append(ids)
    ids = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return ids


def build_sample_cache(rows: list[dict[str, str]], folds: dict[str, int], manifest: dict[str, str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    if SAMPLE_CACHE.exists() and not args.rebuild_sample_cache:
        with np.load(SAMPLE_CACHE) as cache:
            if int(cache["query_count"][0]) == len(rows):
                print(f"decoder reuse sample cache: {SAMPLE_CACHE}", flush=True)
                return {key: cache[key] for key in ("features", "target", "fold")}
    cache: dict[str, ImageData] = {}
    xs, ys, sample_folds = [], [], []
    for index, row in enumerate(rows, start=1):
        data = load_image(row["image_id"], manifest[row["image_id"]], cache)
        valid = data.valid_pixels > 0
        target = data.counts[:, int(row["target_class"])] / np.maximum(data.valid_pixels, 1.0)
        ids = sampled_ids(target, valid, stable_seed(row["query_id"]), args.pos_samples, args.neg_samples, args.mixed_samples)
        features, _ = node_features(data, SCORE_DIR / f"{row['query_id']}_{SCALE}.npz", ids)
        xs.append(features.astype(np.float16))
        ys.append(target[ids].astype(np.float16))
        sample_folds.append(np.full(len(ids), folds[row["image_id"]], dtype=np.int8))
        if index % 500 == 0:
            print(f"decoder sample cache {index}/{len(rows)}", flush=True)
    output = {"features": np.concatenate(xs), "target": np.concatenate(ys), "fold": np.concatenate(sample_folds)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(SAMPLE_CACHE, **output, query_count=np.asarray([len(rows)], dtype=np.int32))
    print(f"decoder wrote sample cache nodes={len(output['target'])}", flush=True)
    return output


def pixel_metric(data: ImageData, class_id: int, predicted: np.ndarray) -> dict[str, float]:
    tp = float(data.counts[predicted, class_id].sum())
    pred_area = float(data.valid_pixels[predicted].sum())
    gt_area = float(data.counts[:, class_id].sum())
    fp, fn = pred_area - tp, gt_area - tp
    denom = 2.0 * tp + fp + fn
    dice = 2.0 * tp / denom if denom else 1.0
    union = tp + fp + fn
    return {"dice": dice, "miou": tp / union if union else 1.0, "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 1.0}


def predict(model: Decoder, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(features), 8192):
            x = torch.from_numpy((features[start:start + 8192] - mean) / std).to(device)
            result.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(result)


def train_fold(
    fold: int, sample_cache: dict[str, np.ndarray], test_rows: list[dict[str, str]], manifest: dict[str, str],
    args: argparse.Namespace, device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache: dict[str, ImageData] = {}
    train_mask = sample_cache["fold"] != fold
    x = sample_cache["features"][train_mask].astype(np.float32)
    y = sample_cache["target"][train_mask].astype(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.maximum(std, 1e-5)
    dataset = TensorDataset(torch.from_numpy((x - mean) / std), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = Decoder(x.shape[1]).to(device)
    positives = float(y.sum())
    pos_weight = float(np.clip((len(y) - positives) / max(positives, 1.0), 1.0, 20.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            prob = torch.sigmoid(logits)
            dice_loss = 1.0 - (2.0 * (prob * batch_y).sum() + 1.0) / (prob.sum() + batch_y.sum() + 1.0)
            loss = bce(logits, batch_y) + args.dice_weight * dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        print(f"decoder fold={fold} epoch={epoch + 1}/{args.epochs} loss={loss_sum / max(len(loader), 1):.4f}", flush=True)
    rows = []
    for row in test_rows:
        data = load_image(row["image_id"], manifest[row["image_id"]], cache)
        features, valid = node_features(data, SCORE_DIR / f"{row['query_id']}_{SCALE}.npz")
        probability = predict(model, features, mean, std, device)
        metric = pixel_metric(data, int(row["target_class"]), (probability >= args.threshold) & valid)
        rows.append({"fold": fold, "query_id": row["query_id"], "image_id": row["image_id"], "target_class": row["target_class"], "prompt_quality": row["prompt_quality"], "threshold": args.threshold, "PixelDice": metric["dice"], "Pixel_mIoU": metric["miou"], "PixelPrecision": metric["precision"], "PixelRecall": metric["recall"], "status": "ok"})
    checkpoint = {"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std, "feature_dim": int(x.shape[1]), "scale": SCALE, "token_file": TOKEN_FILE, "threshold": args.threshold}
    return rows, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos_samples", type=int, default=48)
    parser.add_argument("--neg_samples", type=int, default=144)
    parser.add_argument("--mixed_samples", type=int, default=32)
    parser.add_argument("--fold", type=int, default=-1, help="Debug: only run one test fold; -1 runs all five.")
    parser.add_argument("--rebuild_sample_cache", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase E requires CUDA for this training configuration")
    torch.manual_seed(2026)
    device = torch.device("cuda")
    manifest_rows = read_csv(MANIFEST)
    manifest_frame = {row["image_id"]: row["gt_mask_path"] for row in manifest_rows}
    image_folds = {row["image_id"]: int(row["fold"]) for row in manifest_rows}
    base_rows = [row for row in read_csv(METRICS) if row["scale"] == SCALE and row["status"] == "ok"]
    folds = {row["image_id"]: image_folds[row["image_id"]] for row in base_rows}
    if len(base_rows) != 3711 or len(set(row["query_id"] for row in base_rows)) != 3711:
        raise RuntimeError("unexpected small-scale query rows")
    sample_cache = build_sample_cache(base_rows, folds, manifest_frame, args)
    requested = sorted(set(folds.values())) if args.fold < 0 else [args.fold]
    output: list[dict[str, object]] = []
    checkpoints = []
    for fold in requested:
        test_rows = [row for row in base_rows if folds[row["image_id"]] == fold]
        rows, checkpoint = train_fold(fold, sample_cache, test_rows, manifest_frame, args, device)
        output.extend(rows)
        checkpoints.append(checkpoint)
        print(f"decoder fold={fold} complete queries={len(rows)}", flush=True)
    if args.fold < 0:
        output.sort(key=lambda row: (int(row["fold"]), str(row["query_id"])))
        write_csv(OUT / "mlp_decoder_oof_predictions.csv", output)
        summary = [{"n": len(output), **{key: float(np.mean([float(row[key]) for row in output])) for key in ("PixelDice", "Pixel_mIoU", "PixelPrecision", "PixelRecall")}}]
        write_csv(OUT / "mlp_decoder_results.csv", summary)
        torch.save({"fold_checkpoints": checkpoints, "args": vars(args)}, OUT / "mlp_decoder_oof_models.pt")
        validation = {"passed": len(output) == 3711, "queries": len(output), "folds": requested, "device": torch.cuda.get_device_name(0), "summary": summary[0]}
        (OUT / "mlp_decoder_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        (PHASE_DIR / "report.md").write_text("# Phase E MLP Mask Decoder\n\n" + json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
