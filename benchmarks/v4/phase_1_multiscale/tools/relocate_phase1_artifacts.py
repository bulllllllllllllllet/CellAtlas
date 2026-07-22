#!/usr/bin/env python3
"""Update absolute paths inside retained Phase 1 artifacts after a root relocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def replace(value: object, mappings: dict[str, str]) -> object:
    if isinstance(value, str):
        for old, new in mappings.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace(item, mappings) for item in value]
    if isinstance(value, dict):
        return {key: replace(item, mappings) for key, item in value.items()}
    return value


def rewrite_parquet(path: Path, columns: tuple[str, ...], mappings: dict[str, str]) -> None:
    table = pq.read_table(path)
    for column in columns:
        index = table.schema.get_field_index(column)
        if index < 0:
            raise KeyError(f"{path} lacks expected column {column}")
        values = [replace(value, mappings) for value in table.column(column).to_pylist()]
        table = table.set_column(index, column, pa.array(values, type=table.schema.field(column).type))
    temporary = path.with_suffix(".relocating.parquet")
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mapping", action="append", required=True, metavar="OLD=NEW")
    args = parser.parse_args()
    mappings = dict(item.split("=", 1) for item in args.mapping)
    root = args.root.resolve()
    rewrite_parquet(root / "data/tiled_gt_cohort200_20260716_011804/tiled_gt_manifest.parquet", ("tiled_gt_path",), mappings)
    rewrite_parquet(root / "data/tiled_patch_index_cohort200_20260716_013248/patch_index_10x_tiled_gt.parquet", ("gt_path",), mappings)
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text())
        path.write_text(json.dumps(replace(payload, mappings), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
