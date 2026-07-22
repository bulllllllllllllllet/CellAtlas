import math
import unittest

from benchmarks.v4.phase_6_mask_decoder.src.checkpoint_policy import (
    best_checkpoint_pointer,
    build_pareto_report,
)


GATES = {
    "pixel_macro_dice_min": 0.70,
    "pixel_micro_dice_min": 0.75,
}
SOFT_TARGETS = {
    "region_macro_dice_min": 0.80,
    "unprompted_macro_dice_min": 0.78,
    "boundary_f1_min": 0.28,
    "prompt_conflict_episode_rate_max": 0.07,
}
REFERENCE = {"pixel_macro_dice": 0.71, "pixel_micro_dice": 0.80}


def row(epoch, pixel, region, unprompted, boundary, conflict, pixel_micro=0.82):
    return {
        "epoch": epoch,
        "checkpoint_path": f"checkpoint_epoch_{epoch:03d}.pth",
        "pixel_macro_dice": pixel,
        "pixel_micro_dice": pixel_micro,
        "region_macro_dice": region,
        "unprompted_macro_dice": unprompted,
        "boundary_f1": boundary,
        "prompt_conflict_episode_rate": conflict,
    }


class CheckpointPolicyTests(unittest.TestCase):
    def test_hard_gates_exclude_violation_and_nonfinite_metrics(self):
        report = build_pareto_report([
            row(0, 0.71, 0.81, 0.79, 0.29, 0.08),
            row(1, math.nan, 0.81, 0.79, 0.29, 0.06),
        ], GATES, SOFT_TARGETS, REFERENCE)
        self.assertEqual(report["status"], "pareto_frontier")
        self.assertIn("prompt_conflict_episode_rate_max", report["evaluations"][0]["soft_warnings"])
        self.assertIn("finite_metrics", report["evaluations"][1]["violations"])

    def test_pareto_frontier_removes_dominated_checkpoint(self):
        report = build_pareto_report([
            row(0, 0.71, 0.81, 0.79, 0.29, 0.06),
            row(1, 0.72, 0.82, 0.80, 0.30, 0.05),
            row(2, 0.73, 0.81, 0.79, 0.31, 0.05),
        ], GATES, SOFT_TARGETS, REFERENCE)
        self.assertEqual(report["eligible_epochs"], [0, 1, 2])
        self.assertEqual([item["epoch"] for item in report["frontier"]], [1, 2])
        self.assertEqual(best_checkpoint_pointer(report)["epoch"], 2)

    def test_missing_gate_is_rejected(self):
        incomplete = dict(GATES)
        incomplete.pop("pixel_micro_dice_min")
        with self.assertRaises(KeyError):
            build_pareto_report([row(0, 0.71, 0.81, 0.79, 0.29, 0.06)], incomplete, SOFT_TARGETS, REFERENCE)

    def test_soft_warning_does_not_block_dice_eligible_checkpoint(self):
        report = build_pareto_report([
            row(0, 0.72, 0.70, 0.68, 0.20, 0.20),
        ], GATES, SOFT_TARGETS, REFERENCE)
        pointer = best_checkpoint_pointer(report)
        self.assertEqual(pointer["status"], "eligible_checkpoint")
        self.assertEqual(pointer["epoch"], 0)
        self.assertEqual(len(pointer["soft_warnings"]), 4)

    def test_any_pixel_dice_gain_promotes_micro_improvement(self):
        report = build_pareto_report([
            row(0, 0.709, 0.81, 0.79, 0.29, 0.06, pixel_micro=0.805),
        ], GATES, SOFT_TARGETS, REFERENCE)
        pointer = best_checkpoint_pointer(report)
        self.assertTrue(pointer["any_pixel_dice_improved"])
        self.assertLess(pointer["pixel_macro_dice_gain"], 0)
        self.assertGreater(pointer["pixel_micro_dice_gain"], 0)

    def test_noninferiority_requires_bounded_losses_and_one_gain(self):
        policy = {
            "pixel_macro_dice_margin": 0.001,
            "pixel_micro_dice_margin": 0.001,
            "require_any_pixel_dice_improved": True,
        }
        report = build_pareto_report([
            row(0, 0.7095, 0.81, 0.79, 0.29, 0.06, pixel_micro=0.8002),
            row(1, 0.7089, 0.81, 0.79, 0.29, 0.06, pixel_micro=0.8010),
            row(2, 0.7098, 0.81, 0.79, 0.29, 0.06, pixel_micro=0.7998),
        ], GATES, SOFT_TARGETS, REFERENCE, policy)
        self.assertEqual(report["eligible_epochs"], [0])
        self.assertIn("pixel_macro_dice_noninferior", report["evaluations"][1]["violations"])
        self.assertIn("any_pixel_dice_improved", report["evaluations"][2]["violations"])
        self.assertEqual(report["noninferiority"], policy)


if __name__ == "__main__":
    unittest.main()
