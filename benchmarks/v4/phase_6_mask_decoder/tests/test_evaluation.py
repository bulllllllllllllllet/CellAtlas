import math
import unittest
from pathlib import Path

import torch
import pandas as pd

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import (
    binary_boundary,
    binary_counts,
    boundary_f1,
    dice_from_counts,
    summarize_episode_rows,
)
from benchmarks.v4.phase_6_mask_decoder.tools.evaluate_visualize_joint_pixel import (
    cache_mismatch_is_fatal,
    mismatch_quality,
    summarize_threshold_grid,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_cache_mismatch_gate_only_applies_to_upstream_reference(self):
        self.assertTrue(cache_mismatch_is_fatal(None, 1))
        self.assertFalse(cache_mismatch_is_fatal(None, 0))
        self.assertFalse(cache_mismatch_is_fatal(Path("joint_checkpoint.pth"), 62))

    def test_threshold_grid_selects_best_macro_above_micro_floor(self):
        rows = []
        for target_class, prompt_size in ((0, "point"), (1, "large")):
            row = {"target_class": target_class, "prompt_size": prompt_size}
            for model in ("baseline", "joint"):
                for tag, counts in (("0p400", (8, 2, 2)), ("0p500", (7, 1, 3))):
                    tp, fp, fn = counts
                    prefix = f"{model}_pixel_threshold_{tag}"
                    row.update({
                        f"{prefix}_tp": tp, f"{prefix}_fp": fp, f"{prefix}_fn": fn,
                        f"{prefix}_dice": float(2 * tp / (2 * tp + fp + fn)),
                    })
            rows.append(row)
        summary = summarize_threshold_grid(pd.DataFrame(rows), (0.4, 0.5), 0.7)
        self.assertEqual(summary["models"]["joint"]["selected_by_macro"]["threshold"], 0.4)
        self.assertEqual(set(summary["models"]["joint"]["thresholds"]["0.400"]["by_target_class"]), {"0", "1"})

    def test_tiny_pure_mismatch_is_recorded_but_not_high_purity_region(self):
        assignment = torch.zeros(1, 2, 2, 2)
        assignment[:, 1] = 1.0
        assignment[:, 0, 0, 0] = 2.0
        batch = {
            "pixel_gt": torch.zeros(1, 2, 2, dtype=torch.long),
            "all_prompted_regions": torch.tensor([[True, False]]),
        }
        high, prompted, maximum = mismatch_quality(
            {"assignment": assignment}, batch, torch.tensor([[True, False]]), 255, 2
        )
        self.assertEqual(high, 0)
        self.assertEqual(prompted, 1)
        self.assertEqual(maximum, 1.0)

    def test_binary_counts_are_per_episode(self):
        prediction = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool)
        truth = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
        valid = torch.ones_like(truth)
        counts = binary_counts(prediction, truth, valid)
        self.assertEqual(counts["tp"].tolist(), [1, 1])
        self.assertEqual(counts["fp"].tolist(), [1, 0])
        self.assertEqual(counts["fn"].tolist(), [0, 1])

    def test_dice_empty_is_nan(self):
        self.assertTrue(math.isnan(float(dice_from_counts(0, 0, 0))))
        self.assertAlmostEqual(float(dice_from_counts(2, 1, 1)), 2 / 3)

    def test_boundary_f1_is_one_for_exact_match(self):
        mask = torch.zeros(1, 8, 8, dtype=torch.bool); mask[:, 2:6, 2:6] = True
        valid = torch.ones_like(mask)
        self.assertTrue(binary_boundary(mask, valid).any())
        result = boundary_f1(mask, mask, valid, tolerance=2)
        self.assertAlmostEqual(float(result["boundary_f1"][0]), 1.0)

    def test_summary_excludes_empty_unprompted_targets(self):
        rows = []
        for positive in (1, 0):
            row = {"m_boundary_f1": 1.0}
            for scope in ("region", "unprompted_region", "pixel"):
                row.update({f"m_{scope}_tp": positive, f"m_{scope}_fp": 0, f"m_{scope}_fn": 0, f"m_{scope}_positive": positive})
            rows.append(row)
        summary = summarize_episode_rows(rows, ("m",))
        query = summary["models"]["m"]["unprompted_region"]
        self.assertEqual(query["evaluable_episodes"], 1)
        self.assertEqual(query["episodes_without_unprompted_target"], 1)
        self.assertEqual(query["macro_dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
