from __future__ import annotations

import unittest

import torch

from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder


class PromptTransferTest(unittest.TestCase):
    def test_refactored_local_path_is_exact(self):
        torch.manual_seed(17)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0).eval()
        decoder = ContextAwareMaskDecoder(
            prompt, dim=16, heads=4, graph_layers=1, neighbours=2, dropout=0.0
        ).eval()
        batch = {
            "fine_tokens": torch.randn(2, 9, 16),
            "region_xy": torch.rand(2, 9, 2),
            "region_area": torch.softmax(torch.randn(2, 9), dim=1),
            "fine_active": torch.ones(2, 9, dtype=torch.bool),
            "positive_tokens": torch.randn(2, 3, 16),
            "negative_tokens": torch.randn(2, 2, 16),
            "positive_xy": torch.rand(2, 3, 2),
            "negative_xy": torch.rand(2, 2, 2),
            "positive_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "negative_mask": torch.ones(2, 2, dtype=torch.bool),
            "prompt_size_id": torch.tensor([0, 2]),
        }
        direct_prompt = prompt(batch)
        split_prompt = prompt.match_regions(batch, prompt.encode_prompt_task(batch))
        for key in ("logits", "task_token", "task_aware_tokens", "q_positive", "q_negative"):
            torch.testing.assert_close(direct_prompt[key], split_prompt[key], rtol=0, atol=0)
        direct_decoder = decoder(batch)
        split_decoder = decoder.refine(batch, split_prompt)
        for key in ("logits", "initial_logits", "logit_delta", "decoder_tokens"):
            torch.testing.assert_close(direct_decoder[key], split_decoder[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
