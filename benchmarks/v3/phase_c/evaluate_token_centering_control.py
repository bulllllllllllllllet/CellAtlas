# 作用：在分层抽样 query 上比较原始与去公共均值 token，验证 cosine score 饱和是否为系统性归一化问题。

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gdph_v2.pret_utils import safe_l2_normalize
from benchmarks.v3.phase_c.run_multiscale_baseline import METRICS_PATH, MULTISCALE_ROOT, SCORE_DIR, ranking_metrics, read_csv

REPORT = Path(__file__).resolve().parent / "token_centering_control_report.json"
SCALE = "small"


def score(tokens: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    positive_score = np.max(tokens @ tokens[positive].T, axis=1)
    if not len(negative):
        return positive_score
    negative_score = np.max(tokens @ tokens[negative].T, axis=1)
    return positive_score - 0.5 * negative_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_quality", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = [row for row in read_csv(METRICS_PATH) if row["scale"] == SCALE and row["status"] == "ok"]
    by_quality: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_quality[row["prompt_quality"]].append(row)
    rng = np.random.default_rng(args.seed)
    selected = []
    for quality, items in sorted(by_quality.items()):
        ids = rng.choice(len(items), size=min(args.per_quality, len(items)), replace=False)
        selected.extend(items[int(index)] for index in ids)

    cache = {}
    output = defaultdict(list)
    for index, row in enumerate(selected, start=1):
        image_id = row["image_id"]
        if image_id not in cache:
            directory = MULTISCALE_ROOT / image_id / SCALE
            validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            source = Path(validation["source_dir"])
            records = read_csv(directory / "superpixels.csv")
            labels = np.asarray([int(float(item.get("gt_majority_label", 255))) for item in records], dtype=np.int16)
            valid = (labels >= 0) & (labels < 12) & (np.asarray([float(item.get("valid_fraction", 0.0)) for item in records]) > 0)
            variants = {}
            for name, filename in (("image", "tokens_image_only.npy"), ("enhanced", "tokens_image_cell_reg_texture_cellstats.npy")):
                original = safe_l2_normalize(np.asarray(np.load(source / filename, mmap_mode="r"), dtype=np.float32), axis=1)
                centered = safe_l2_normalize(original - original[valid].mean(axis=0, keepdims=True), axis=1)
                variants[name] = (original, centered)
            cache[image_id] = labels, valid, variants
        labels, valid, variants = cache[image_id]
        target = labels == int(row["target_class"])
        with np.load(SCORE_DIR / f"{row['query_id']}_{SCALE}.npz") as prompt:
            positive = prompt["positive_prompt_segments"].astype(np.int64)
            negative = prompt["negative_prompt_segments"].astype(np.int64)
        for variant, pair in variants.items():
            for transform, tokens in zip(("original", "centered"), pair, strict=True):
                values = score(tokens, positive, negative)
                ap, auc = ranking_metrics(target[valid], values[valid])
                output[(row["prompt_quality"], variant, transform)].append((ap, auc, float(values[valid].std()), float(values[valid].mean())))
        if index % 50 == 0:
            print(f"token_centering {index}/{len(selected)}", flush=True)

    summary = []
    for (quality, variant, transform), values in sorted(output.items()):
        array = np.asarray(values, dtype=np.float64)
        summary.append({"prompt_quality": quality, "variant": variant, "transform": transform, "n": len(values), "mAP": float(array[:, 0].mean()), "AUROC": float(array[:, 1].mean()), "score_std": float(array[:, 2].mean()), "score_mean": float(array[:, 3].mean())})
    payload = {"sampled_queries": len(selected), "per_quality": args.per_quality, "summary": summary}
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
