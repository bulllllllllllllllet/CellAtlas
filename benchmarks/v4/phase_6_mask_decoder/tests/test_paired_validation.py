import unittest

import pandas as pd

from benchmarks.v4.phase_6_mask_decoder.src.paired_validation import (
    paired_episode_bootstrap,
    paired_point_estimates,
)


class PairedValidationTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        return pd.DataFrame({
            "baseline_pixel_tp": [8, 6, 5, 9],
            "baseline_pixel_fp": [2, 3, 2, 1],
            "baseline_pixel_fn": [2, 1, 3, 1],
            "joint_pixel_tp": [9, 7, 6, 9],
            "joint_pixel_fp": [1, 2, 1, 1],
            "joint_pixel_fn": [1, 0, 2, 1],
        })

    def test_point_estimates_preserve_pairing(self):
        result = paired_point_estimates(self.frame())
        self.assertEqual(result["episodes"], 4)
        self.assertGreater(result["macro_dice_difference"], 0.0)
        self.assertGreater(result["micro_dice_difference"], 0.0)

    def test_bootstrap_is_deterministic_and_reports_rules(self):
        first = paired_episode_bootstrap(self.frame(), samples=200, seed=7)
        second = paired_episode_bootstrap(self.frame(), samples=200, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first["point_rule_passed"])
        self.assertEqual(first["bootstrap_samples"], 200)

    def test_rejects_unevaluable_episode(self):
        frame = self.frame(); frame.loc[0, ["baseline_pixel_tp", "baseline_pixel_fp", "baseline_pixel_fn"]] = 0
        with self.assertRaisesRegex(ValueError, "finite pixel Dice"):
            paired_episode_bootstrap(frame, samples=10)


if __name__ == "__main__":
    unittest.main()
