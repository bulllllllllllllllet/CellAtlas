import unittest

import torch

from benchmarks.v4.phase_6_mask_decoder.src.conflict_policy import (
    ABSTAIN_REASON,
    build_inference_response,
    conflict_free_training_batch,
    conflict_stress_rows,
)


class ConflictPolicyTests(unittest.TestCase):
    def test_training_filter_preserves_metadata_and_removes_conflicts(self):
        output = {
            "online_prompt_conflicts": torch.tensor([[False, False], [True, False], [False, False]]),
            "logits": torch.arange(6).reshape(3, 2).float(),
        }
        batch = {"patch_id": ["a", "b", "c"], "target_class": torch.tensor([1, 2, 3])}
        keep, selected_output, selected_batch = conflict_free_training_batch(output, batch)
        torch.testing.assert_close(keep, torch.tensor([True, False, True]))
        self.assertEqual(selected_batch["patch_id"], ["a", "c"])
        torch.testing.assert_close(selected_output["logits"], output["logits"][[0, 2]])

    def test_stress_rows_preserve_each_conflict_occurrence(self):
        output = {
            "online_prompt_conflicts": torch.tensor([[False, True], [False, True]]),
            "online_positive_slot_indices": torch.tensor([[1, -1], [1, -1]]),
            "online_negative_slot_indices": torch.tensor([[1], [1]]),
        }
        batch = {
            "episode_index": torch.tensor([7, 7]),
            "patch_id": ["p", "p"], "wsi_id": ["w", "w"],
            "target_class": torch.tensor([2, 2]), "prompt_size": ["point", "point"],
            "positive_mask": torch.tensor([[True, False], [True, False]]),
            "negative_mask": torch.ones(2, 1, dtype=torch.bool),
            "positive_xy": torch.tensor([[[0.1, 0.2], [0.0, 0.0]], [[0.1, 0.2], [0.0, 0.0]]]),
            "negative_xy": torch.tensor([[[0.3, 0.4]], [[0.3, 0.4]]]),
        }
        rows = conflict_stress_rows(output, batch, epoch=1, rank=0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode_index"], 7)
        self.assertEqual(rows[0]["conflict_slot_indices"], [1])

    def test_inference_abstains_without_returning_mask(self):
        output = {
            "online_prompt_conflicts": torch.tensor([[False, True], [False, False]]),
            "pixel_probability": torch.rand(2, 4, 4),
        }
        abstained = build_inference_response(output, 0)
        self.assertEqual(abstained["status"], "abstain")
        self.assertEqual(abstained["reason"], ABSTAIN_REASON)
        self.assertIsNone(abstained["pixel_probability"])
        accepted = build_inference_response(output, 1)
        self.assertEqual(accepted["status"], "ok")
        torch.testing.assert_close(accepted["pixel_probability"], output["pixel_probability"][1])


if __name__ == "__main__":
    unittest.main()
