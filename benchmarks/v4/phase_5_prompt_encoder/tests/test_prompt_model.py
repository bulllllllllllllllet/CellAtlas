import unittest
from types import SimpleNamespace
from collections import Counter

import torch
import numpy as np

from benchmarks.v4.phase_5_prompt_encoder.src.losses import metric_counts, metrics_from_counts, prompt_region_loss
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import EpisodeBalancedSampler


class PromptModelTest(unittest.TestCase):
    def batch(self, batch_size=2):
        torch.manual_seed(4)
        target = torch.zeros(batch_size, 64, dtype=torch.long)
        target[:, :8] = 1
        target[:, -2:] = 255
        prompted = torch.zeros(batch_size, 64, dtype=torch.bool)
        prompted[:, :3] = True
        return {
            "fine_tokens": torch.randn(batch_size, 64, 256),
            "region_xy": torch.rand(batch_size, 64, 2),
            "region_area": torch.full((batch_size, 64), 1 / 64),
            "positive_tokens": torch.randn(batch_size, 8, 256),
            "negative_tokens": torch.randn(batch_size, 3, 256),
            "positive_xy": torch.rand(batch_size, 8, 2),
            "negative_xy": torch.rand(batch_size, 3, 2),
            "positive_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]] * batch_size, dtype=torch.bool),
            "negative_mask": torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.bool),
            "prompt_size_id": torch.tensor([1] * batch_size),
            "binary_target": target,
            "prompted_regions": prompted,
        }

    def test_forward_loss_backward_and_phase6_contract(self):
        batch = self.batch()
        model = PromptRegionModel(dropout=0.0)
        output = model(batch)
        self.assertEqual(output["logits"].shape, (2, 64))
        self.assertEqual(output["task_token"].shape, (2, 256))
        self.assertEqual(output["task_aware_tokens"].shape, (2, 64, 256))
        loss, parts = prompt_region_loss(output, batch["binary_target"], batch["prompted_regions"], 255, 1.0, 0.2, 0.2)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"balanced_bce", "dice_loss", "ranking_loss"})
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_unprompted_recall_excludes_prompt_slots(self):
        batch = self.batch(1)
        logits = torch.full((1, 64), -1.0)
        logits[:, :3] = 1.0
        counts = {name: float(value) for name, value in metric_counts(logits, batch["binary_target"], batch["prompted_regions"], 255).items()}
        metrics = metrics_from_counts(counts)
        self.assertEqual(counts["unprompted_positive"], 5)
        self.assertEqual(metrics["unprompted_target_recall"], 0.0)

    def test_episode_sampler_respects_requested_marginals(self):
        rows = []
        for class_id in range(3):
            for size in ("point", "small", "large"):
                for group in ("inside", "boundary"):
                    rows.extend({"target_class": class_id, "prompt_size": size, "sampling_group": group} for _ in range(2))
        dataset = SimpleNamespace(
            target_classes=np.asarray([row["target_class"] for row in rows]),
            prompt_sizes=np.asarray([row["prompt_size"] for row in rows]),
            sampling_groups=np.asarray([row["sampling_group"] for row in rows]),
        )
        sampler = EpisodeBalancedSampler(dataset, {"point": .4, "small": .35, "large": .25}, {"inside": .6, "boundary": .4}, (0, 1, 2), 7, epoch_size=60000)
        selected = [rows[i] for i in sampler]
        sizes = Counter(row["prompt_size"] for row in selected)
        classes = Counter(row["target_class"] for row in selected)
        self.assertLess(abs(sizes["point"] / len(selected) - .4), .01)
        self.assertLess(max(classes.values()) - min(classes.values()), 500)
        self.assertEqual(sampler.empty_buckets, [])


if __name__ == "__main__":
    unittest.main()
