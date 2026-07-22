from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features


class CellFeatureStore:
    """Strict random access over a Phase3 shard-level feature manifest."""

    def __init__(self, manifest_path: str | Path, expected_rows: int, max_open_shards: int = 8):
        manifest = pd.read_parquet(manifest_path).sort_values("start").reset_index(drop=True)
        required = {"shard_path", "start", "end", "rows"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"cell feature manifest misses {sorted(required - set(manifest.columns))}")
        starts = manifest["start"].to_numpy(dtype=np.int64)
        ends = manifest["end"].to_numpy(dtype=np.int64)
        counts = manifest["rows"].to_numpy(dtype=np.int64)
        if len(manifest) == 0 or starts[0] != 0 or ends[-1] != int(expected_rows):
            raise ValueError("cell feature manifest does not cover the complete tile index")
        if len(manifest) > 1 and not np.array_equal(starts[1:], ends[:-1]):
            raise ValueError("cell feature manifest has a gap or overlap")
        if not np.array_equal(ends - starts, counts):
            raise ValueError("cell feature manifest rows disagree with start/end")
        paths = [Path(value) for value in manifest["shard_path"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cell feature shard missing: {missing[0]}")
        self.manifest = manifest
        self.ends = ends
        self.paths = paths
        self.max_open_shards = int(max_open_shards)
        self.cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()

    def get(self, source_index: int, patch_id: str) -> tuple[np.ndarray, int]:
        source_index = int(source_index)
        shard_index = int(np.searchsorted(self.ends, source_index, side="right"))
        if shard_index >= len(self.manifest):
            raise IndexError(source_index)
        row_ref = self.manifest.iloc[shard_index]
        path = self.paths[shard_index]
        table = self.cache.get(path)
        if table is None:
            table = pd.read_parquet(
                path, columns=["source_index", "patch_id", "cells", "reg_features", "total_cell_count"]
            )
            self.cache[path] = table
            if len(self.cache) > self.max_open_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        offset = source_index - int(row_ref["start"])
        row = table.iloc[offset]
        if int(row.source_index) != source_index or str(row.patch_id) != str(patch_id):
            raise RuntimeError(
                f"cell feature identity mismatch at {source_index}: {row.patch_id} != {patch_id}"
            )
        cells = np.stack(row.cells).astype(np.float32) if len(row.cells) else np.empty((0, 4), np.float32)
        reg = np.stack(row.reg_features).astype(np.float32) if len(row.reg_features) else np.empty((0, 64), np.float32)
        return encode_xcell_features(cells, reg), int(row.total_cell_count)
