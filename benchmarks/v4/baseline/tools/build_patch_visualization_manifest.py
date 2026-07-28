#!/usr/bin/env python3
"""Select explicit frozen occurrences for a small patch-visualization replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.baseline.common import sha256_path, validate_episode_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--occurrence-order", type=int, action="append", required=True)
    parser.add_argument(
        "--selection-rule",
        required=True,
        help="Human-readable frozen rule used to choose the explicit occurrences.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    output = args.output_root / f"patch_visualization_manifest_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    source = pd.read_parquet(args.source_manifest)
    if len(set(args.occurrence_order)) != len(args.occurrence_order):
        raise ValueError("occurrence orders must be unique")
    selected = source[source["occurrence_order"].isin(args.occurrence_order)].copy()
    if len(selected) != len(args.occurrence_order):
        found = set(selected["occurrence_order"].astype(int))
        raise ValueError(f"missing occurrence orders: {sorted(set(args.occurrence_order) - found)}")
    selected["source_occurrence_order"] = selected["occurrence_order"].astype(np.int64)
    selected = selected.set_index("occurrence_order").loc[args.occurrence_order].reset_index()
    selected["occurrence_order"] = np.arange(len(selected), dtype=np.int64)
    audit = validate_episode_manifest(selected, "val")

    output.mkdir(parents=True)
    manifest = output / f"episode_manifest_{args.timestamp}.parquet"
    selected.to_parquet(manifest, index=False)
    report = {
        "timestamp": args.timestamp,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_path(args.source_manifest),
        "selection_rule": args.selection_rule,
        "requested_source_occurrence_orders": args.occurrence_order,
        "manifest": str(manifest),
        "manifest_sha256": sha256_path(manifest),
        "audit": audit,
    }
    (output / f"metadata_{args.timestamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
