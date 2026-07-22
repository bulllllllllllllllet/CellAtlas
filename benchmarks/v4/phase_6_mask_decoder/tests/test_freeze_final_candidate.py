import argparse
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.v4.phase_6_mask_decoder.tools.freeze_final_candidate import build_manifest


class FreezeFinalCandidateTests(unittest.TestCase):
    def args(self, root: Path, *, split: str = "val", macro: float = 0.734) -> argparse.Namespace:
        checkpoint = root / "candidate.pth"
        config = root / "config.yaml"
        stress = root / "stress.parquet"
        summary = root / "summary.json"
        checkpoint.write_bytes(b"checkpoint")
        config.write_text("pixel_macro_dice_min: 0.72\n")
        stress.write_bytes(b"stress")
        summary.write_text(json.dumps({
            "episode_count": 4000,
            "split": split,
            "test_used": split == "test",
            "models": {
                "baseline": {"pixel": {"macro_dice": 0.732, "micro_dice": 0.804}},
                "joint": {"pixel": {"macro_dice": macro, "micro_dice": 0.807}},
            },
            "joint_prompt_conflict_episodes": 284,
            "joint_prompt_conflict_episode_rate": 0.071,
        }))
        return argparse.Namespace(
            checkpoint=checkpoint,
            config=config,
            validation_summary=summary,
            validation_stress_set=stress,
            timestamp="20260722_000000",
            pixel_macro_floor=0.72,
            pixel_micro_floor=0.7987,
            pixel_threshold=0.5,
        )

    def test_freeze_records_passing_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(self.args(Path(directory)))
        self.assertEqual(manifest["status"], "frozen_pre_test")
        self.assertFalse(manifest["test_evaluated"])
        self.assertEqual(manifest["selection"]["hard_gates"]["pixel_macro_dice_min"], 0.72)
        self.assertTrue(all(manifest["selection"]["checks"].values()))

    def test_freeze_rejects_test_metrics_and_failed_macro_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "validation-only"):
                build_manifest(self.args(Path(directory), split="test"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "failed frozen selection"):
                build_manifest(self.args(Path(directory), macro=0.719))


if __name__ == "__main__":
    unittest.main()
