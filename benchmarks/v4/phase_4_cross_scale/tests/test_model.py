#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from benchmarks.v4.phase_4_cross_scale.src.model import (
    AttentionHierarchicalBlock,
    CrossScaleModel,
    gather_children,
    gather_parents,
)


class ModelTests(unittest.TestCase):
    @staticmethod
    def _hierarchical_batch(dim=8):
        return {
            "fine_tokens": torch.randn(1, 3, dim),
            "middle_tokens": torch.randn(1, 3, dim),
            "coarse_tokens": torch.randn(1, 3, dim),
            "fine_middle_edge_index": torch.tensor([[[0], [1], [2]]]),
            "fine_middle_edge_weight": torch.ones(1, 3, 1),
            "middle_coarse_edge_index": torch.tensor([[[0], [1], [2]]]),
            "middle_coarse_edge_weight": torch.ones(1, 3, 1),
        }

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

    def test_hierarchical_up_calls_only_upward_messages(self):
        model = CrossScaleModel("hierarchical_up", dim=8, num_classes=2)
        block = model.blocks[0]
        batch = self._hierarchical_batch()
        with (
            patch.object(block.up_fm, "forward_from_children", wraps=block.up_fm.forward_from_children) as up_fm,
            patch.object(block.up_mc, "forward_from_children", wraps=block.up_mc.forward_from_children) as up_mc,
            patch.object(block.down_cm, "forward_from_parents", wraps=block.down_cm.forward_from_parents) as down_cm,
            patch.object(block.down_mf, "forward_from_parents", wraps=block.down_mf.forward_from_parents) as down_mf,
        ):
            model(batch)
        self.assertEqual(up_fm.call_count, 1)
        self.assertEqual(up_mc.call_count, 1)
        self.assertEqual(down_cm.call_count, 0)
        self.assertEqual(down_mf.call_count, 0)

    def test_hierarchical_bidir_calls_both_directions(self):
        model = CrossScaleModel("hierarchical_bidir", dim=8, num_classes=2)
        block = model.blocks[0]
        batch = self._hierarchical_batch()
        with (
            patch.object(block.up_fm, "forward_from_children", wraps=block.up_fm.forward_from_children) as up_fm,
            patch.object(block.up_mc, "forward_from_children", wraps=block.up_mc.forward_from_children) as up_mc,
            patch.object(block.down_cm, "forward_from_parents", wraps=block.down_cm.forward_from_parents) as down_cm,
            patch.object(block.down_mf, "forward_from_parents", wraps=block.down_mf.forward_from_parents) as down_mf,
        ):
            model(batch)
        self.assertEqual(up_fm.call_count, 1)
        self.assertEqual(up_mc.call_count, 1)
        self.assertEqual(down_cm.call_count, 1)
        self.assertEqual(down_mf.call_count, 1)

    def test_up_and_bidir_diverge_with_identical_weights(self):
        torch.manual_seed(7)
        up = CrossScaleModel("hierarchical_up", dim=8, num_classes=2)
        bidir = CrossScaleModel("hierarchical_bidir", dim=8, num_classes=2)
        bidir.load_state_dict(up.state_dict())
        batch = self._hierarchical_batch()
        up_logits = up(batch)
        bidir_logits = bidir(batch)
        self.assertFalse(torch.allclose(up_logits, bidir_logits))

    def test_sparse_parent_attention_is_finite_and_content_dependent(self):
        torch.manual_seed(11)
        block = AttentionHierarchicalBlock(8)
        fine = torch.randn(1, 3, 8, requires_grad=True)
        middle = torch.randn(1, 3, 8, requires_grad=True)
        coarse = torch.randn(1, 3, 8, requires_grad=True)
        index = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
        weight = torch.tensor([[[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]])
        updated, _, _ = block.forward_bidir(fine, middle, coarse, index, weight, index, weight)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertIsNotNone(block.down_mf.last_attention_entropy)
        self.assertGreater(float(block.down_mf.last_attention_entropy), 0.0)
        updated.square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in block.down_mf.parameters() if parameter.grad is not None
        ), 0.0)


if __name__ == "__main__":
    unittest.main()
