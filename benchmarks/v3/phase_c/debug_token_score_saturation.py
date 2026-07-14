# 作用：诊断指定 query 的 score 饱和来源，比较颜色、细胞、base/enhanced token，并检查底层 cell-reg 特征是否塌缩。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import safe_l2_normalize
from benchmarks.v3.phase_c.run_multiscale_baseline import METRICS_PATH, MULTISCALE_ROOT, SCORE_DIR, ranking_metrics, read_csv

PRET_ROOT = MULTISCALE_ROOT.parent
MANIFEST = PRET_ROOT / "data_manifest_v3.csv"
REPORT = Path(__file__).resolve().parent / "token_score_saturation_report.json"
QUANTILES = (0.0, 0.01, 0.02, 0.1, 0.5, 0.9, 0.98, 0.99, 1.0)


def distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES), strict=True)},
        "fraction_ge_0p90": float(np.mean(values >= 0.90)),
        "fraction_ge_0p95": float(np.mean(values >= 0.95)),
        "fraction_ge_0p98": float(np.mean(values >= 0.98)),
    }


def pairwise_cosine(tokens: np.ndarray, seed: int, pairs: int = 100000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    count = len(tokens)
    left = rng.integers(0, count, size=min(pairs, max(count * 10, 1)))
    right = rng.integers(0, count, size=len(left))
    keep = left != right
    values = np.sum(tokens[left[keep]] * tokens[right[keep]], axis=1)
    return distribution(values)


def retrieval_report(tokens: np.ndarray, positive: np.ndarray, negative: np.ndarray, target: np.ndarray, valid: np.ndarray, seed: int) -> tuple[dict[str, object], np.ndarray]:
    score_pos = np.max(tokens @ tokens[positive].T, axis=1)
    score_neg = np.max(tokens @ tokens[negative].T, axis=1) if len(negative) else np.zeros(len(tokens), dtype=np.float32)
    score = score_pos - 0.5 * score_neg if len(negative) else score_pos
    ap, auc = ranking_metrics(target[valid], score[valid])
    return {
        "score": distribution(score[valid]),
        "score_target_mean": float(score[valid & target].mean()),
        "score_nontarget_mean": float(score[valid & ~target].mean()),
        "mAP_majority": ap,
        "AUROC_majority": auc,
        "pairwise_cosine": pairwise_cosine(tokens[valid], seed),
    }, score


def score_variant(path: Path, positive: np.ndarray, negative: np.ndarray, target: np.ndarray, valid: np.ndarray, seed: int) -> tuple[dict[str, object], np.ndarray]:
    tokens = safe_l2_normalize(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32), axis=1)
    original, score = retrieval_report(tokens, positive, negative, target, valid, seed)
    feature_std = tokens.std(axis=0)
    mean = tokens[valid].mean(axis=0, keepdims=True)
    centered = safe_l2_normalize(tokens - mean, axis=1)
    std = np.maximum(tokens[valid].std(axis=0, keepdims=True), 1e-4)
    zscore = safe_l2_normalize((tokens - mean) / std, axis=1)
    centered_report, _ = retrieval_report(centered, positive, negative, target, valid, seed + 100)
    zscore_report, _ = retrieval_report(zscore, positive, negative, target, valid, seed + 200)
    return {
        "path": str(path),
        "shape": list(tokens.shape),
        **original,
        "centered_retrieval": centered_report,
        "zscore_retrieval": zscore_report,
        "mean_dimension_std": float(feature_std.mean()),
        "near_constant_dimensions": int(np.sum(feature_std < 1e-5)),
    }, score


def cell_feature_audit(path: Path, seed: int) -> dict[str, object]:
    values = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(values), size=min(100000, len(values)), replace=False)
    sample = np.asarray(values[ids], dtype=np.float32)
    normalized = safe_l2_normalize(sample, axis=1)
    centered = sample - sample.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    explained = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    return {
        "path": str(path),
        "shape": list(values.shape),
        "finite": bool(np.isfinite(sample).all()),
        "mean_dimension_std": float(sample.std(axis=0).mean()),
        "min_dimension_std": float(sample.std(axis=0).min()),
        "near_constant_dimensions": int(np.sum(sample.std(axis=0) < 1e-5)),
        "pairwise_cosine": pairwise_cosine(normalized, seed + 1),
        "explained_variance_top1": float(explained[:1].sum()),
        "explained_variance_top10": float(explained[:10].sum()),
        "effective_rank": float(np.exp(-np.sum(explained[explained > 0] * np.log(explained[explained > 0])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_id", default="1512244-15-HE-DX1_c0_msp_noisy_001")
    parser.add_argument("--scale", default="small", choices=("small", "medium", "large"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metric = next(row for row in read_csv(METRICS_PATH) if row["query_id"] == args.query_id and row["scale"] == args.scale)
    image_id, class_id = metric["image_id"], int(metric["target_class"])
    directory = MULTISCALE_ROOT / image_id / args.scale
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    source = Path(validation["source_dir"])
    rows = read_csv(directory / "superpixels.csv")
    labels = np.asarray([int(float(row.get("gt_majority_label", 255))) for row in rows], dtype=np.int16)
    valid = (labels >= 0) & (labels < 12) & (np.asarray([float(row.get("valid_fraction", 0.0)) for row in rows]) > 0)
    target = labels == class_id
    with np.load(SCORE_DIR / f"{args.query_id}_{args.scale}.npz") as saved:
        positive = saved["positive_prompt_segments"].astype(np.int64)
        negative = saved["negative_prompt_segments"].astype(np.int64)
        saved_score = saved["score_final"].astype(np.float32)
    variants = {
        "image_only_color_geometry": source / "tokens_image_only.npy",
        "cell_reg_only": source / "tokens_cell_reg.npy",
        "image_cell_equal": source / "tokens_image_cell_reg.npy",
        "base_cellw0p5": source / "tokens_image_cell_reg_cellw0p5.npy",
        "base_cellw2p0": source / "tokens_image_cell_reg_cellw2p0.npy",
        "enhanced_texture_cellstats": source / "tokens_image_cell_reg_texture_cellstats.npy",
    }
    variant_reports = {}
    recomputed = {}
    for index, (name, path) in enumerate(variants.items()):
        variant_reports[name], recomputed[name] = score_variant(path, positive, negative, target, valid, args.seed + index)
    manifest = next(row for row in read_csv(MANIFEST) if row["image_id"] == image_id)
    enhanced = np.asarray(np.load(variants["enhanced_texture_cellstats"], mmap_mode="r"), dtype=np.float32)
    block_norms = {
        "base_274": distribution(np.linalg.norm(enhanced[:, :274], axis=1)),
        "texture_30": distribution(np.linalg.norm(enhanced[:, 274:304], axis=1)),
        "cellstats_5": distribution(np.linalg.norm(enhanced[:, 304:309], axis=1)),
    }
    payload = {
        "query_id": args.query_id,
        "image_id": image_id,
        "scale": args.scale,
        "target_class": class_id,
        "prompt_quality": metric["prompt_quality"],
        "positive_prompt_segments": positive.tolist(),
        "negative_prompt_segments": negative.tolist(),
        "saved_score": distribution(saved_score[valid]),
        "saved_vs_recomputed_enhanced_max_abs": float(np.max(np.abs(saved_score - recomputed["enhanced_texture_cellstats"]))),
        "enhanced_block_norms": block_norms,
        "variants": variant_reports,
        "underlying_cell_reg": cell_feature_audit(Path(manifest["cell_feature_reg_path"]), args.seed),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
