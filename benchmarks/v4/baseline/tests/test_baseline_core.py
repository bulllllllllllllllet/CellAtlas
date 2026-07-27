from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v4.baseline.adapters.base import EpisodeRequest, Prediction
from benchmarks.v4.baseline.adapters.fixed_slic_adapter import FixedSLICAdapter
from benchmarks.v4.baseline.common import binary_metric_row, exact_occurrence_alignment, summarize_metrics, validate_episode_manifest
from benchmarks.v4.baseline.tools.build_baseline_report import markdown_table
from benchmarks.v4.baseline.tools.build_gt_guided_wsi_tasks import sample_pixel_center, stable_seed
from benchmarks.v4.baseline.tools.build_prompt_geometry import hard_region_representatives
from benchmarks.v4.baseline.integrations.sam_external import read_he_patch_white_padded


class BaselineCoreTest(unittest.TestCase):
    def request(self) -> EpisodeRequest:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :16] = (210, 80, 90); image[:, 16:] = (60, 180, 90)
        return EpisodeRequest(
            occurrence_id="val_000000", image=image,
            positive_points=np.asarray([[5, 5]], np.float32),
            negative_points=np.asarray([[27, 27]], np.float32), prompt_type="point",
        )

    def test_prediction_contract_rejects_non_probability(self):
        prediction = Prediction(np.full((2, 2), 1.1, np.float32), np.zeros((2, 2), bool))
        with self.assertRaises(ValueError):
            prediction.validate((2, 2))

    def test_fixed_slic_is_deterministic_and_finite(self):
        adapter = FixedSLICAdapter({"n_segments": 8, "compactness": 10, "temperature": 8, "threshold": 0.5})
        first = adapter.timed_predict(self.request()); second = adapter.timed_predict(self.request())
        np.testing.assert_array_equal(first.binary_mask, second.binary_mask)
        np.testing.assert_allclose(first.probability, second.probability)
        self.assertTrue(np.isfinite(first.probability).all())

    def test_failed_empty_mask_remains_in_denominator(self):
        metric = binary_metric_row(np.zeros((4, 4), bool), np.eye(4, dtype=bool), np.ones((4, 4), bool))
        frame = pd.DataFrame([metric | {
            "status": "failed", "prompt_size": "point", "target_class": 1,
            "latency_ms": 1.0, "peak_memory_mb": 0.0,
        }])
        summary = summarize_metrics(frame)
        self.assertEqual(summary["coverage"], 0.0)
        self.assertEqual(summary["pooled_pixel_dice"], 0.0)

    def test_manifest_and_alignment_contract(self):
        row = {
            "occurrence_id": "val_000000", "occurrence_order": 0, "episode_index": 3,
            "split": "val", "patch_id": "p", "wsi_id": "w", "patient_id": "patient",
            "target_class": 1, "prompt_size": "point",
            "positive_points_10x": "[[1,2]]", "negative_points_10x": "[[3,4]]",
            "positive_box_10x": None, "x_10x": 0, "y_10x": 0, "width_10x": 8,
            "height_10x": 8, "x_level0": 0, "y_level0": 0, "width_level0": 16,
            "height_level0": 16, "wsi_path": "/readonly/wsi", "gt_path": "/readonly/gt",
            "source_region_ids": "[1]",
        }
        frame = pd.DataFrame([row])
        audit = validate_episode_manifest(frame, "val")
        self.assertEqual(audit["rows"], 1)
        self.assertEqual(exact_occurrence_alignment([frame, frame.copy()]), ["val_000000"])
        altered = frame.copy(); altered.loc[0, "occurrence_id"] = "other"
        with self.assertRaises(ValueError):
            exact_occurrence_alignment([frame, altered])

    def test_markdown_table_has_no_optional_dependency(self):
        rendered = markdown_table(pd.DataFrame([{"Method": "Fixed-SLIC", "Dice": 0.5}]))
        self.assertIn("Fixed-SLIC", rendered)
        self.assertIn("| Method", rendered)

    def test_hard_region_representatives_are_max_probability_member_pixel_centers(self):
        assignment = np.asarray([
            [[0.90, 0.10, 0.70, 0.10], [0.10, 0.80, 0.70, 0.20], [0.70, 0.60, 0.10, 0.10]],
            [[0.05, 0.80, 0.10, 0.10], [0.70, 0.10, 0.10, 0.10], [0.10, 0.10, 0.10, 0.10]],
            [[0.05, 0.10, 0.20, 0.75], [0.20, 0.10, 0.20, 0.70], [0.20, 0.30, 0.80, 0.80]],
            [[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]],
        ], dtype=np.float32)
        hard = assignment.argmax(0)
        points, modes = hard_region_representatives(assignment, np.asarray([1, 2]))
        np.testing.assert_array_equal(points, np.asarray([[1.5, 0.5], [2.5, 2.5]], np.float32))
        self.assertEqual(modes, ["hard_member_max_probability", "hard_member_max_probability"])
        for point, slot in zip(points, [1, 2], strict=True):
            x, y = np.floor(point).astype(int)
            self.assertEqual(int(hard[y, x]), slot)
        points, modes = hard_region_representatives(assignment, np.asarray([3]))
        np.testing.assert_array_equal(points, np.asarray([[0.5, 0.5]], np.float32))
        self.assertEqual(modes, ["soft_max_probability_empty_hard_slot"])

    def test_gt_guided_pixel_sampling_is_deterministic_and_inside_mask(self):
        mask = np.asarray([[False, True, False], [True, False, True]])
        seed = stable_seed(20260726, "wsi", 7, "patch", "positive")
        self.assertEqual(seed, stable_seed(20260726, "wsi", 7, "patch", "positive"))
        point = sample_pixel_center(mask, seed)
        self.assertEqual(point, sample_pixel_center(mask, seed))
        x, y = np.floor(point).astype(int)
        self.assertTrue(mask[y, x])
        self.assertIsNone(sample_pixel_center(np.zeros((2, 2), dtype=bool), seed))

    def test_wsi_sam_context_uses_white_padding_without_shifting_center(self):
        source = np.zeros((4, 4, 3), dtype=np.uint8)
        source[:] = (10, 20, 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tif"
            Image.fromarray(source, "RGB").save(path)
            context = read_he_patch_white_padded(path, -2, -2, 8, 8, 8)
        self.assertEqual(context.shape, (8, 8, 3))
        np.testing.assert_array_equal(context[:2, :], 255)
        np.testing.assert_array_equal(context[:, :2], 255)
        np.testing.assert_array_equal(context[2:6, 2:6], source)
        np.testing.assert_array_equal(context[6:, :], 255)
        np.testing.assert_array_equal(context[:, 6:], 255)


if __name__ == "__main__":
    unittest.main()
