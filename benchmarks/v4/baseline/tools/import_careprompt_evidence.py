#!/usr/bin/env python3
"""Convert immutable J3/J5 audit evidence to the unified baseline row schema."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.baseline.common import append_jsonl, atomic_json, new_output_directory, sha256_path, summarize_metrics, timestamp, validate_episode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--evidence-metrics", type=Path, required=True)
    parser.add_argument("--model-prefix", choices=("baseline", "joint"), required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--expected-episodes", type=int, default=4000)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    manifest = pd.read_parquet(args.episode_manifest).sort_values("occurrence_order").reset_index(drop=True)
    validate_episode_manifest(manifest, args.split)
    evidence = pd.read_parquet(args.evidence_metrics)
    if len(manifest) != args.expected_episodes or len(evidence) != args.expected_episodes:
        raise ValueError("manifest/evidence count differs from expected episodes")
    if evidence["episode_index"].duplicated().any():
        raise ValueError("evidence has duplicate episode_index")
    evidence = evidence.set_index("episode_index")
    missing = sorted(set(manifest["episode_index"]) - set(evidence.index))
    if missing:
        raise ValueError(f"evidence misses episode indices {missing[:10]}")
    rows = []
    prefix = args.model_prefix
    for record in manifest.to_dict("records"):
        source = evidence.loc[int(record["episode_index"])]
        for name in ("patch_id", "wsi_id", "target_class", "prompt_size"):
            if str(record[name]) != str(source[name]):
                raise RuntimeError(f"evidence identity mismatch episode={record['episode_index']} field={name}")
        rows.append({
            **{name: record[name] for name in (
                "occurrence_id", "occurrence_order", "episode_index", "patch_id", "wsi_id",
                "patient_id", "target_class", "prompt_size",
            )},
            **{name: int(source[f"{prefix}_pixel_{name}"]) for name in ("tp", "fp", "fn", "tn", "positive", "valid")},
            "episode_dice": float(source[f"{prefix}_pixel_dice"]),
            "boundary_f1": float(source[f"{prefix}_boundary_f1"]),
            "boundary_evaluable": bool(source[f"{prefix}_boundary_evaluable"]),
            "status": "completed", "latency_ms": np.nan, "peak_memory_mb": np.nan,
            "candidate_info": json.dumps({"source": "immutable_internal_audit_evidence", "prefix": prefix}),
            "rank": int(source.get("rank", 0)),
        })
    frame = pd.DataFrame(rows)
    # Exact count recomputation prevents trusting stale serialized Dice fields.
    denominator = 2 * frame.tp + frame.fp + frame.fn
    recomputed = np.divide(2 * frame.tp, denominator, out=np.full(len(frame), np.nan), where=denominator > 0)
    if not np.allclose(recomputed, frame.episode_dice, equal_nan=True, atol=1e-12):
        raise RuntimeError("evidence Dice does not equal recomputed TP/FP/FN Dice")
    output = new_output_directory(args.output_root, args.method_name, stamp)
    completed = output / f"completed_{stamp}.jsonl"; failures = output / f"failures_{stamp}.jsonl"
    failures.open("x", encoding="utf-8").close()
    for occurrence_id in frame["occurrence_id"]:
        append_jsonl(completed, {"occurrence_id": str(occurrence_id), "status": "imported_completed_evidence"})
    metrics = output / f"episode_metrics_{stamp}.parquet"; frame.to_parquet(metrics, index=False)
    summary = summarize_metrics(frame) | {
        "timestamp": stamp, "split": args.split, "test_used": args.split == "test",
        "method": args.method_name, "evidence": str(args.evidence_metrics),
        "evidence_sha256": sha256_path(args.evidence_metrics), "manifest": str(args.episode_manifest),
        "manifest_sha256": sha256_path(args.episode_manifest),
        "efficiency_metrics_available": False,
    }
    atomic_json(output / f"summary_{stamp}.json", summary)
    atomic_json(output / f"run_metadata_{stamp}.json", {
        "timestamp": stamp, "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "evidence": str(args.evidence_metrics), "evidence_sha256": sha256_path(args.evidence_metrics),
        "manifest": str(args.episode_manifest), "manifest_sha256": sha256_path(args.episode_manifest),
        "model_prefix": args.model_prefix, "method": args.method_name,
    })
    (output / f"command_{stamp}.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": str(metrics), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
