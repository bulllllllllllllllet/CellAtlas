#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from benchmarks.v4.phase_4_cross_scale.src.model import CrossScaleModel, gather_children, gather_parents


class ModelTests(unittest.TestCase):
    def test_gather_parents_weight_sum(self):
        parent = torch.randn(2, 4, 8)
        edge_index = torch.tensor([[[0, 1, -1, -1], [2, -1, -1, -1]], [[1, 3, -1, -1], [0, 2, -1, -1]]])
        edge_weight = torch.tensor([[[0.7, 0.3, 0, 0], [1.0, 0, 0, 0]], [[0.5, 0.5, 0, 0], [0.25, 0.75, 0, 0]]])
        out = gather_parents(parent, edge_index, edge_weight)
        self.assertEqual(out.shape, (2, 2, 8))
        # first batch first child = 0.7*p0 + 0.3*p1
        expected = 0.7 * parent[0, 0] + 0.3 * parent[0, 1]
        self.assertTrue(torch.allclose(out[0, 0], expected, atol=1e-5))

    def test_gather_children_roundtrip_mass(self):
        child = torch.ones(1, 3, 4)
        edge_index = torch.tensor([[[0, -1], [0, -1], [1, -1]]])
        edge_weight = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        parent = gather_children(child, edge_index, edge_weight, n_parent=2)
        self.assertEqual(parent.shape, (1, 2, 4))
        self.assertTrue(torch.allclose(parent[0, 0], torch.ones(4)))
        self.assertTrue(torch.allclose(parent[0, 1], torch.ones(4)))

    def test_variants_forward(self):
        batch = {
            "fine_tokens": torch.randn(2, 64, 256),
            "middle_tokens": torch.randn(2, 64, 256),
            "coarse_tokens": torch.randn(2, 64, 256),
            "fine_middle_edge_index": torch.randint(0, 64, (2, 64, 4)),
            "fine_middle_edge_weight": torch.softmax(torch.randn(2, 64, 4), dim=-1),
            "middle_coarse_edge_index": torch.randint(0, 64, (2, 64, 4)),
            "middle_coarse_edge_weight": torch.softmax(torch.randn(2, 64, 4), dim=-1),
        }
        for variant in CrossScaleModel.VARIANTS:
            model = CrossScaleModel(variant)
            logits = model(batch)
            self.assertEqual(logits.shape, (2, 64, 12), variant)
            self.assertTrue(torch.isfinite(logits).all(), variant)


if __name__ == "__main__":
    unittest.main()
