import unittest

import torch

from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.losses import decoder_loss
from benchmarks.v4.phase_6_mask_decoder.src.model import (
    ContextAwareMaskDecoder,
    knn_adjacency,
    project_region_probabilities,
)


def batch(batch_size=2, regions=8, dim=16):
    active = torch.ones(batch_size, regions, dtype=torch.bool)
    active[:, -1] = False
    target = torch.randint(0, 2, (batch_size, regions))
    target[:, 0] = 1
    target[:, 1] = 0
    target[~active] = 255
    return {
        "fine_tokens": torch.randn(batch_size, regions, dim),
        "fine_active": active,
        "region_xy": torch.rand(batch_size, regions, 2),
        "region_area": torch.rand(batch_size, regions),
        "positive_tokens": torch.randn(batch_size, 3, dim),
        "negative_tokens": torch.randn(batch_size, 2, dim),
        "positive_xy": torch.rand(batch_size, 3, 2),
        "negative_xy": torch.rand(batch_size, 2, 2),
        "positive_mask": torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool),
        "negative_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "prompt_size_id": torch.tensor([0, 2]),
        "prompted_regions": torch.zeros(batch_size, regions, dtype=torch.bool),
        "binary_target": target,
    }


class MaskDecoderTests(unittest.TestCase):
    def test_knn_never_connects_inactive_keys(self):
        value = batch()
        adjacency = knn_adjacency(value["region_xy"], value["fine_active"], 3)
        self.assertEqual(adjacency.shape, (2, 8, 8))
        self.assertFalse(adjacency[:, :-1, -1].any())
        self.assertTrue(adjacency[:, -1, -1].all())

    def test_zero_initialized_decoder_matches_phase5(self):
        torch.manual_seed(3)
        value = batch()
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = ContextAwareMaskDecoder(prompt, dim=16, heads=4, graph_layers=2, dropout=0.0)
        model.eval()
        with torch.no_grad():
            expected = prompt(value)["logits"]
            output = model(value)
        torch.testing.assert_close(output["logits"], expected)
        self.assertTrue((output["logit_delta"] == 0).all())

    def test_residual_has_zero_active_mean(self):
        value = batch()
        model = ContextAwareMaskDecoder(
            PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0),
            dim=16, heads=4, graph_layers=1, dropout=0.0,
        )
        torch.nn.init.normal_(model.delta_head[-1].weight)
        output = model(value)
        active = value["fine_active"]
        mean = (output["logit_delta"] * active).sum(1) / active.sum(1)
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-6, rtol=0)
        self.assertTrue((output["logit_delta"][~active] == 0).all())

    def test_decoder_loss_is_finite_and_backpropagates(self):
        value = batch()
        model = ContextAwareMaskDecoder(
            PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0),
            dim=16, heads=4, graph_layers=1, dropout=0.0,
        )
        output = model(value)
        loss, parts = decoder_loss(
            output, value["binary_target"], value["prompted_regions"], 255, 1.0, 0.2, 0.2, 0.1
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("boundary_loss", parts)
        loss.backward()
        self.assertIsNotNone(model.delta_head[-1].weight.grad)

    def test_soft_projection(self):
        assignment = torch.tensor([[[[1.0, 0.25]], [[0.0, 0.75]]]])
        logits = torch.tensor([[20.0, -20.0]])
        result = project_region_probabilities(assignment, logits)
        self.assertEqual(result["pixel_probability"].shape, (1, 1, 2))
        self.assertGreater(float(result["pixel_probability"][0, 0, 0]), 0.99)
        self.assertAlmostEqual(float(result["pixel_probability"][0, 0, 1]), 0.25, places=4)

    def test_bfloat16_saturated_projection_has_finite_logits(self):
        assignment = torch.ones(1, 1, 2, 2, dtype=torch.bfloat16)
        logits = torch.tensor([[100.0]], dtype=torch.bfloat16)
        result = project_region_probabilities(assignment, logits)
        self.assertEqual(result["pixel_probability"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["pixel_logits"]).all())


if __name__ == "__main__":
    unittest.main()
