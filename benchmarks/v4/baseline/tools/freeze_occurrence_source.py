#!/usr/bin/env python3
"""Freeze occurrence identities from an audited validation evaluation artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.v4.baseline.common import atomic_json, new_output_directory, sha256_path, timestamp


IDENTITY_COLUMNS = ["episode_index", "patch_id", "wsi_id", "target_class", "prompt_size"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-metrics", type=Path, required=True)
    parser.add_argument("--audit-metadata", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--expected-occurrences", type=int, default=4000)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    metadata = json.loads(args.audit_metadata.read_text(encoding="utf-8"))
    audit = metadata.get("summary", metadata)
    if audit.get("split") != args.split or bool(audit.get("test_used", False)):
        raise RuntimeError("audit metadata is not validation-only")
    source = pd.read_parquet(args.audit_metrics)
    missing = set(IDENTITY_COLUMNS) - set(source)
    if missing:
        raise ValueError(f"audit metrics miss identity columns {sorted(missing)}")
    if len(source) != args.expected_occurrences:
        raise ValueError(f"occurrence count {len(source)} != {args.expected_occurrences}")
    frozen = source[IDENTITY_COLUMNS].copy()
    if frozen.isna().any().any():
        raise ValueError("occurrence identity contains null values")
    if not frozen["prompt_size"].isin(["point", "small", "large"]).all():
        raise ValueError("occurrence identity contains unsupported prompt sizes")
    output = new_output_directory(args.output_root, "occurrence_contract", stamp)
    path = output / f"audited_occurrences_{stamp}.parquet"
    frozen.to_parquet(path, index=False)
    record = {
        "timestamp": stamp, "split": args.split, "test_used": False,
        "occurrences": len(frozen), "unique_episode_indices": int(frozen["episode_index"].nunique()),
        "repeated_occurrences": int(frozen["episode_index"].duplicated().sum()),
        "prompt_size_counts": frozen["prompt_size"].value_counts().sort_index().to_dict(),
        "target_class_counts": {str(k): int(v) for k, v in frozen["target_class"].value_counts().sort_index().items()},
        "source_metrics": str(args.audit_metrics), "source_metrics_sha256": sha256_path(args.audit_metrics),
        "source_metadata": str(args.audit_metadata), "source_metadata_sha256": sha256_path(args.audit_metadata),
        "occurrence_source": str(path), "occurrence_source_sha256": sha256_path(path),
        "identity_columns": IDENTITY_COLUMNS,
    }
    atomic_json(output / f"metadata_{stamp}.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

