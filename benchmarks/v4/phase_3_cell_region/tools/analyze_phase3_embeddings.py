#!/usr/bin/env python3
"""Compute kNN / retrieval / distance metrics on exported Phase-3 embeddings."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.v4.phase_1_multiscale.src.common import load_config


FOCUS_CLASSES = ("blood", "necrosis", "lymphocyte_aggregate")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-failures-per-class", type=int, default=20)
    parser.add_argument("--retrieval-queries-per-class", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[int(t), int(p)] += 1
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(1).astype(np.float64)
    predicted = matrix.sum(0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    summary = {
        "accuracy": float(tp.sum() / max(support.sum(), 1.0)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1.0)),
        "valid_regions": int(support.sum()),
    }
    rows = [
        {
            "class_id": i,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(class_count)
    ]
    return summary, rows, matrix


def knn_predict(emb: np.ndarray, labels: np.ndarray, k: int, chunk_size: int, device: str = "cpu") -> np.ndarray:
    import torch

    n = emb.shape[0]
    preds = np.empty(n, dtype=np.int16)
    bank = torch.from_numpy(emb).to(device)
    label_t = torch.from_numpy(labels.astype(np.int64)).to(device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        query = bank[start:end]
        sims = query @ bank.T
        for local_i, global_i in enumerate(range(start, end)):
            sims[local_i, global_i] = float("-inf")
        topk = torch.topk(sims, k=min(k, n - 1), dim=1).indices
        top_labels = label_t[topk].cpu().numpy()
        top_sims = sims.gather(1, topk).cpu().numpy()
        for local_i in range(end - start):
            counts = defaultdict(float)
            for lab, sim in zip(top_labels[local_i].tolist(), top_sims[local_i].tolist(), strict=True):
                counts[int(lab)] += 1.0 + 1e-6 * float(sim)
            preds[start + local_i] = max(counts.items(), key=lambda item: item[1])[0]
    return preds


def retrieval_map(
    emb: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    chunk_size: int,
    device: str = "cpu",
    queries_per_class: int = 256,
    seed: int = 20260719,
) -> dict:
    """Class-balanced retrieval mAP on a fixed query subsample.

    Full all-pairs AP over ~250k regions is too expensive for closeout; sample
    up to ``queries_per_class`` queries per class with a fixed seed.
    """
    import torch

    support = np.bincount(labels, minlength=class_count)
    ap_sum = np.zeros(class_count, dtype=np.float64)
    query_count = np.zeros(class_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    query_ids = []
    for class_id in range(class_count):
        ids = np.where(labels == class_id)[0]
        if ids.size <= 1:
            continue
        take = min(queries_per_class, ids.size)
        query_ids.append(rng.choice(ids, size=take, replace=False))
    if not query_ids:
        return {"class_balanced_map": 0.0, "micro_map": 0.0, "per_class": []}
    query_ids = np.concatenate(query_ids)
    bank = torch.from_numpy(emb).to(device)
    labels_np = labels
    for start in range(0, len(query_ids), chunk_size):
        batch = query_ids[start : start + chunk_size]
        sims = (bank[batch] @ bank.T).cpu().numpy()
        for local_i, global_i in enumerate(batch.tolist()):
            label = int(labels_np[global_i])
            positives = support[label] - 1
            if positives <= 0:
                continue
            sims[local_i, global_i] = -np.inf
            order = np.argsort(-sims[local_i])
            hits = labels_np[order] == label
            if not hits.any():
                continue
            ranks = np.nonzero(hits)[0] + 1
            precision_at = (np.arange(1, len(ranks) + 1, dtype=np.float64)) / ranks.astype(np.float64)
            ap_sum[label] += float(precision_at.mean())
            query_count[label] += 1
    per_class = []
    present = []
    for class_id in range(class_count):
        value = float(ap_sum[class_id] / query_count[class_id]) if query_count[class_id] else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "map": value,
                "query_count": int(query_count[class_id]),
                "support": int(support[class_id]),
            }
        )
        if query_count[class_id] > 0:
            present.append(value)
    return {
        "class_balanced_map": float(np.mean(present)) if present else 0.0,
        "micro_map": float(ap_sum.sum() / max(query_count.sum(), 1)),
        "per_class": per_class,
        "queries_per_class_budget": int(queries_per_class),
        "seed": int(seed),
    }


def class_distances(emb: np.ndarray, labels: np.ndarray, class_count: int) -> list[dict]:
    centroids = []
    for class_id in range(class_count):
        mask = labels == class_id
        if not mask.any():
            centroids.append(None)
            continue
        c = emb[mask].mean(axis=0)
        centroids.append(c / max(np.linalg.norm(c), 1e-8))
    rows = []
    for class_id in range(class_count):
        mask = labels == class_id
        if not mask.any() or centroids[class_id] is None:
            rows.append(
                {
                    "class_id": class_id,
                    "support": int(mask.sum()),
                    "mean_intra_cosine_distance": None,
                    "nearest_inter_class_id": None,
                    "nearest_inter_centroid_cosine_distance": None,
                }
            )
            continue
        intra = 1.0 - (emb[mask] @ centroids[class_id])
        nearest_id, nearest_dist = None, None
        for other in range(class_count):
            if other == class_id or centroids[other] is None:
                continue
            dist = float(1.0 - float(np.dot(centroids[class_id], centroids[other])))
            if nearest_dist is None or dist < nearest_dist:
                nearest_id, nearest_dist = other, dist
        rows.append(
            {
                "class_id": class_id,
                "support": int(mask.sum()),
                "mean_intra_cosine_distance": float(intra.mean()),
                "nearest_inter_class_id": nearest_id,
                "nearest_inter_centroid_cosine_distance": nearest_dist,
            }
        )
    return rows


def failure_cases(
    emb: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    patch_ids: np.ndarray,
    region_ids: np.ndarray,
    wsi_ids: np.ndarray,
    class_names: list[str],
    focus_ids: list[int],
    max_failures: int,
) -> list[dict]:
    rows = []
    for class_id in focus_ids:
        fail_idx = np.where((labels == class_id) & (preds != class_id))[0]
        if fail_idx.size == 0:
            continue
        # rank failures by distance to own class centroid among failures
        own = emb[labels == class_id].mean(axis=0)
        own = own / max(np.linalg.norm(own), 1e-8)
        scores = emb[fail_idx] @ own
        order = np.argsort(scores)  # most dissimilar first
        for local in order[:max_failures]:
            idx = int(fail_idx[local])
            # nearest neighbor true label for context
            sims = emb[idx] @ emb.T
            sims[idx] = -np.inf
            nn = int(np.argmax(sims))
            rows.append(
                {
                    "true_class_id": class_id,
                    "true_class_name": class_names[class_id],
                    "pred_class_id": int(preds[idx]),
                    "pred_class_name": class_names[int(preds[idx])],
                    "patch_id": str(patch_ids[idx]),
                    "wsi_id": str(wsi_ids[idx]),
                    "region_id": int(region_ids[idx]),
                    "nn_patch_id": str(patch_ids[nn]),
                    "nn_class_id": int(labels[nn]),
                    "nn_class_name": class_names[int(labels[nn])],
                    "nn_cosine": float(sims[nn]),
                    "own_centroid_cosine": float(scores[local]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_variant(
    name: str,
    emb: np.ndarray,
    labels: np.ndarray,
    patch_ids: np.ndarray,
    region_ids: np.ndarray,
    wsi_ids: np.ndarray,
    class_names: list[str],
    k: int,
    chunk_size: int,
    max_failures: int,
    output: Path,
    device: str = "cpu",
    queries_per_class: int = 256,
    seed: int = 20260719,
) -> dict:
    emb_n = l2_normalize(emb)
    preds = knn_predict(emb_n, labels, k=k, chunk_size=chunk_size, device=device)
    knn_summary, knn_rows, knn_matrix = confusion_metrics(labels, preds, len(class_names))
    for row in knn_rows:
        row["class_name"] = class_names[row["class_id"]]
    retrieval = retrieval_map(
        emb_n,
        labels,
        len(class_names),
        chunk_size=chunk_size,
        device=device,
        queries_per_class=queries_per_class,
        seed=seed,
    )
    for row in retrieval["per_class"]:
        row["class_name"] = class_names[row["class_id"]]
    distances = class_distances(emb_n, labels, len(class_names))
    for row in distances:
        row["class_name"] = class_names[row["class_id"]]
        if row["nearest_inter_class_id"] is not None:
            row["nearest_inter_class_name"] = class_names[row["nearest_inter_class_id"]]
        else:
            row["nearest_inter_class_name"] = None
    focus_ids = [i for i, name_ in enumerate(class_names) if name_ in FOCUS_CLASSES]
    failures = failure_cases(
        emb_n, labels, preds, patch_ids, region_ids, wsi_ids, class_names, focus_ids, max_failures
    )
    np.save(output / f"knn_confusion_{name}.npy", knn_matrix)
    write_csv(
        output / f"knn_per_class_{name}.csv",
        knn_rows,
        ["class_id", "class_name", "precision", "recall", "f1", "support"],
    )
    write_csv(
        output / f"retrieval_per_class_{name}.csv",
        retrieval["per_class"],
        ["class_id", "class_name", "map", "query_count", "support"],
    )
    write_csv(
        output / f"class_distances_{name}.csv",
        distances,
        [
            "class_id",
            "class_name",
            "support",
            "mean_intra_cosine_distance",
            "nearest_inter_class_id",
            "nearest_inter_class_name",
            "nearest_inter_centroid_cosine_distance",
        ],
    )
    write_csv(
        output / f"focus_failures_{name}.csv",
        failures,
        [
            "true_class_id",
            "true_class_name",
            "pred_class_id",
            "pred_class_name",
            "patch_id",
            "wsi_id",
            "region_id",
            "nn_patch_id",
            "nn_class_id",
            "nn_class_name",
            "nn_cosine",
            "own_centroid_cosine",
        ],
    )
    return {
        "knn": knn_summary,
        "retrieval": {
            "class_balanced_map": retrieval["class_balanced_map"],
            "micro_map": retrieval["micro_map"],
        },
        "focus_failure_count": {
            class_names[i]: int(((labels == i) & (preds != i)).sum()) for i in focus_ids
        },
    }


def main():
    args = parse()
    export_dir = args.export_dir
    output = args.output_dir or export_dir
    output.mkdir(parents=True, exist_ok=True)
    p2cfg = load_config(args.phase2_config)
    class_names = [row["name"] for row in p2cfg["data"]["class_map"]]
    labels = np.load(export_dir / "labels.npy")
    patch_ids = np.load(export_dir / "patch_ids.npy", allow_pickle=True)
    region_ids = np.load(export_dir / "region_ids.npy")
    wsi_ids = np.load(export_dir / "wsi_ids.npy", allow_pickle=True)
    meta = json.loads((export_dir / "export_metadata.json").read_text(encoding="utf-8"))
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {
        "k": args.k,
        "device": device,
        "retrieval_queries_per_class": args.retrieval_queries_per_class,
        "seed": args.seed,
        "variants": {},
        "export_metadata": meta,
    }
    for name in ("cell_full", "region_only"):
        emb = np.load(export_dir / f"{name}.npy")
        if emb.shape[0] != labels.shape[0]:
            raise RuntimeError(f"{name} embedding count mismatch: {emb.shape[0]} != {labels.shape[0]}")
        print({"event": "analyze_variant_start", "variant": name, "regions": int(emb.shape[0])}, flush=True)
        summary["variants"][name] = analyze_variant(
            name,
            emb,
            labels,
            patch_ids,
            region_ids,
            wsi_ids,
            class_names,
            args.k,
            args.chunk_size,
            args.max_failures_per_class,
            output,
            device=device,
            queries_per_class=args.retrieval_queries_per_class,
            seed=args.seed,
        )
        print({"event": "analyze_variant_done", "variant": name, **summary["variants"][name]["knn"]}, flush=True)
    cell = summary["variants"]["cell_full"]
    region = summary["variants"]["region_only"]
    summary["comparison"] = {
        "cell_full_minus_region_only_knn_accuracy": cell["knn"]["accuracy"] - region["knn"]["accuracy"],
        "cell_full_minus_region_only_knn_macro_f1": cell["knn"]["macro_f1"] - region["knn"]["macro_f1"],
        "cell_full_minus_region_only_class_balanced_map": (
            cell["retrieval"]["class_balanced_map"] - region["retrieval"]["class_balanced_map"]
        ),
    }
    (output / "embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({"event": "analysis_complete", "output": str(output), **summary["comparison"]}, flush=True)


if __name__ == "__main__":
    main()
