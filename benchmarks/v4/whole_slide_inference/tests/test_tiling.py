from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.v4.whole_slide_inference.src.features import CellFeatureStore
from benchmarks.v4.whole_slide_inference.src.tiling import (
    OverlapAccumulator,
    blend_window,
    build_tile_rows,
    sliding_starts,
    validate_tile_rows,
)


class TilingTest(unittest.TestCase):
    def test_sliding_grid_reaches_last_pixel(self):
        self.assertEqual(sliding_starts(1300, 512, 384), [0, 384, 768, 788])
        rows = build_tile_rows("wsi", "/read/only.tif", 2600, 2200, 2, 512, 384, "val")
        self.assertEqual(validate_tile_rows(rows), (1300, 1100, 2))
        self.assertEqual(len(rows), 12)

    def test_overlap_fusion_reconstructs_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            accumulator = OverlapAccumulator(Path(directory), 20, 24, 0.5)
            window = blend_window(16, 16)
            value = np.full((16, 16), 0.73, np.float32)
            for x, y in ((0, 0), (8, 0), (0, 4), (8, 4)):
                accumulator.add(value, x, y, window)
            probability, mask = accumulator.finalize(chunk_rows=7)
            np.testing.assert_allclose(probability, 0.73, rtol=0, atol=2e-6)
            self.assertTrue(np.all(mask == 255))
            self.assertTrue(np.all(accumulator.weight_sum > 0))

    def test_cell_feature_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "features_0000000_0000000.parquet"
            pd.DataFrame([{
                "source_index": 0, "patch_id": "p0", "total_cell_count": 1,
                "cells": [[0.25, 0.75, 1.0, 2.0]],
                "reg_features": [[0.0] * 64],
            }]).to_parquet(shard, index=False)
            manifest = root / "feature_index.parquet"
            pd.DataFrame([{
                "shard_path": str(shard), "start": 0, "end": 1, "rows": 1,
            }]).to_parquet(manifest, index=False)
            features, total = CellFeatureStore(manifest, 1).get(0, "p0")
            self.assertEqual(features.shape, (1, 74))
            self.assertEqual(total, 1)
            self.assertEqual(float(features[0, 3 + 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
