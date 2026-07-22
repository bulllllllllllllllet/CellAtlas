import unittest

import torch
import torch.nn.functional as F

from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_5_prompt_encoder.src.losses import prompt_region_loss
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import (
    joint_pixel_loss,
    prompt_conflict_margin_loss,
    prompt_safe_geometry_anchor_loss,
    prompt_separation_loss,
)
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import (
    FrozenPromptTeacher,
    FrozenRegionGeometryTeacher,
    JointPromptMaskModel,
    ParentContextAdapter,
    assignment_geometry,
    gather_prompt_tokens,
    online_region_binary_target,
    remap_prompt_tokens,
)


class TinyPhase2(torch.nn.Module):
    def __init__(self, regions=4, dim=16):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.layer4 = torch.nn.Conv2d(3, 3, 1, bias=False)
        with torch.no_grad():
            self.backbone.layer4.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
        self.semantic = torch.nn.Identity()
        self.embedding = torch.nn.Conv2d(3, dim, 1)
        self.assignment = torch.nn.Conv2d(dim, regions, 1)

    def extract_backbone_features(self, image):
        return self.backbone.layer4(image)

    def forward_from_features(self, features, output_size, return_full_assignment=True, return_tokens=True):
        embedding = self.embedding(features)
        assignment_low = self.assignment(embedding).softmax(1)
        mass = assignment_low.sum((2, 3)).clamp_min(1e-6)
        tokens = torch.einsum("bkhw,bdhw->bkd", assignment_low, embedding) / mass.unsqueeze(-1)
        return {
            "assignment_low": assignment_low,
            "assignment": assignment_low,
            "region_tokens": tokens,
            "semantic_logits": features.new_zeros(features.shape[0], 2, *output_size),
        }

    def forward(self, image, return_full_assignment=True, return_tokens=True):
        return self.forward_from_features(
            image, image.shape[-2:], return_full_assignment, return_tokens
        )


def make_batch(batch_size=2, regions=4, cell_dim=4):
    pixel_gt = torch.zeros(batch_size, 8, 8, dtype=torch.long)
    pixel_gt[:, :, 4:] = 1
    fine_tokens = torch.randn(batch_size, regions, 16)
    return {
        "image": torch.randn(batch_size, 3, 8, 8),
        "pixel_gt": pixel_gt,
        "target_class": torch.ones(batch_size, dtype=torch.long),
        "cells": torch.rand(batch_size, 3, cell_dim),
        "cell_valid": torch.ones(batch_size, 3, dtype=torch.bool),
        "total_cell_count": torch.full((batch_size,), 3, dtype=torch.long),
        "fine_active": torch.ones(batch_size, regions, dtype=torch.bool),
        "fine_tokens": fine_tokens,
        "region_xy": torch.rand(batch_size, regions, 2),
        "region_area": torch.full((batch_size, regions), 1.0 / regions),
        "positive_slot_indices": torch.tensor([[0, -1], [1, -1]]),
        "negative_slot_indices": torch.tensor([[2], [3]]),
        "positive_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        "negative_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "positive_tokens": torch.stack([fine_tokens[:, 0], torch.zeros_like(fine_tokens[:, 0])], dim=1),
        "negative_tokens": fine_tokens[:, 2:3],
        "positive_xy": torch.rand(batch_size, 2, 2),
        "negative_xy": torch.rand(batch_size, 1, 2),
        "prompt_size_id": torch.zeros(batch_size, dtype=torch.long),
        "prompted_regions": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
        "binary_target": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
        "middle_tokens": torch.randn(batch_size, regions, 16),
        "coarse_tokens": torch.randn(batch_size, regions, 16),
        "fine_middle_edge_index": torch.arange(regions).view(1, regions, 1).expand(batch_size, -1, -1),
        "fine_middle_edge_weight": torch.ones(batch_size, regions, 1),
        "middle_coarse_edge_index": torch.arange(regions).view(1, regions, 1).expand(batch_size, -1, -1),
        "middle_coarse_edge_weight": torch.ones(batch_size, regions, 1),
    }


class JointPixelTests(unittest.TestCase):
    def test_j7_trains_only_backbone_layer4_and_decoder(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
            train_cell=False, train_prompt=False, train_decoder=True,
            train_parent_context=False, train_backbone_layer4=True,
        )
        output = model(make_batch())
        loss = output["pixel_probability"].mean() + output["logits"].square().mean()
        loss.backward()
        self.assertGreater(float(phase2.backbone.layer4.weight.grad.abs().sum()), 0.0)
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.parameters() if parameter.grad is not None
        ), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in phase2.embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in phase2.assignment.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in cell.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.prompt_model.parameters()))

    def test_j5_only_backbone_layer4_is_trainable(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
            train_cell=False, train_prompt=False, train_decoder=False,
            train_parent_context=False, train_backbone_layer4=True,
        )
        batch = make_batch(); output = model(batch)
        loss, _ = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1,
             "region_aux": 0.25, "region_ranking": 0,
             "assignment_balance": 0.01, "assignment_entropy": 0.001,
             "assignment_compactness": 0.01},
            0.2,
        )
        loss.backward()
        self.assertGreater(float(phase2.backbone.layer4.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in phase2.embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in phase2.assignment.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in cell.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))

    def test_parent_context_is_zero_equivalent_and_trainable(self):
        batch = make_batch()
        adapter = ParentContextAdapter(16)
        fine = torch.randn(2, 4, 16)
        contextual = adapter(fine, batch)
        torch.testing.assert_close(contextual, fine, rtol=0, atol=0)
        contextual.square().mean().backward()
        self.assertIsNotNone(adapter.gate.grad)
        self.assertGreater(float(adapter.gate.grad.abs()), 0.0)

    def test_j4_parent_context_freezes_all_existing_modules(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
            train_cell=False, train_prompt=False, train_decoder=False,
            train_parent_context=True,
        )
        batch = make_batch(); output = model(batch)
        torch.testing.assert_close(output["online_contextual_tokens"], output["online_fused_tokens"], rtol=0, atol=0)
        loss, _ = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1,
             "region_aux": 0.25, "region_ranking": 0,
             "assignment_balance": 0, "assignment_entropy": 0,
             "assignment_compactness": 0},
            0.2,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in phase2.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in cell.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parent_context.parameters() if parameter.grad is not None
        ), 0.0)

    def test_geometry_is_normalized_and_differentiable(self):
        assignment = torch.rand(2, 4, 5, 6, requires_grad=True).softmax(1)
        xy, area = assignment_geometry(assignment)
        self.assertEqual(xy.shape, (2, 4, 2)); self.assertEqual(area.shape, (2, 4))
        self.assertTrue(((xy >= 0) & (xy <= 1)).all())
        torch.testing.assert_close(area.sum(1), torch.ones(2))
        (xy.sum() + area.sum()).backward()
        self.assertIsNotNone(assignment.grad if assignment.is_leaf else assignment.grad_fn)

    def test_prompt_gather_masks_padding(self):
        tokens = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
        indices = torch.tensor([[2, -1], [1, 3]])
        valid = indices >= 0
        gathered = gather_prompt_tokens(tokens, indices, valid)
        torch.testing.assert_close(gathered[0, 0], tokens[0, 2])
        self.assertTrue((gathered[0, 1] == 0).all())

    def test_online_prompt_remap_uses_coordinates_and_merges_duplicates(self):
        tokens = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        region_xy = torch.tensor([[[0.25, 0.5], [0.75, 0.5]]], requires_grad=True)
        prompt_xy = torch.tensor([[[0.25, 0.25], [0.25, 0.75], [0.0, 0.0]]])
        valid = torch.tensor([[True, True, False]])
        prompted, slots, membership, weights, soft = remap_prompt_tokens(
            tokens, region_xy, prompt_xy, valid
        )
        torch.testing.assert_close(slots, torch.tensor([[0, 0, -1]]))
        torch.testing.assert_close(membership, torch.tensor([[True, False]]))
        self.assertTrue((prompted[0, 2] == 0).all())
        self.assertTrue((weights[0, 2] == 0).all())
        self.assertTrue((soft[0, 2] == 0).all())
        prompted.sum().backward()
        self.assertIsNotNone(tokens.grad); self.assertIsNotNone(region_xy.grad)

    def test_online_region_target_uses_current_majority_class(self):
        assignment = torch.zeros(1, 2, 2, 4)
        assignment[:, 0, :, :2] = 1.0; assignment[:, 1, :, 2:] = 1.0
        gt = torch.tensor([[[1, 1, 0, 0], [1, 0, 0, 1]]])
        target, purity = online_region_binary_target(
            assignment, gt, torch.tensor([1]), torch.ones(1, 2, dtype=torch.bool), 2, 255
        )
        torch.testing.assert_close(target, torch.tensor([[1, 0]]))
        torch.testing.assert_close(purity, torch.tensor([[0.75, 0.75]]))

    def test_prompt_separation_penalizes_shared_online_region(self):
        batch = {
            "positive_mask": torch.tensor([[True]]),
            "negative_mask": torch.tensor([[True]]),
        }
        shared = {
            "online_positive_prompt_soft_weights": torch.tensor([[[1.0, 0.0]]]),
            "online_negative_prompt_soft_weights": torch.tensor([[[1.0, 0.0]]]),
        }
        separate = {
            "online_positive_prompt_soft_weights": torch.tensor([[[1.0, 0.0]]]),
            "online_negative_prompt_soft_weights": torch.tensor([[[0.0, 1.0]]]),
        }
        self.assertGreater(float(prompt_separation_loss(shared, batch)), 0.9)
        self.assertEqual(float(prompt_separation_loss(separate, batch)), 0.0)

    def test_prompt_conflict_margin_targets_shared_hard_slot(self):
        region_xy = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]], requires_grad=True)
        output = {
            "online_region_xy": region_xy,
            "online_positive_slot_indices": torch.tensor([[0]]),
            "online_negative_slot_indices": torch.tensor([[0]]),
        }
        batch = {
            "positive_xy": torch.tensor([[[0.25, 0.5]]]),
            "negative_xy": torch.tensor([[[0.35, 0.5]]]),
            "positive_mask": torch.tensor([[True]]),
            "negative_mask": torch.tensor([[True]]),
        }
        loss, pairs = prompt_conflict_margin_loss(output, batch)
        self.assertEqual(int(pairs), 1)
        self.assertAlmostEqual(float(loss.detach()), 0.3, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(region_xy.grad).all())

    def test_safe_geometry_anchor_excludes_conflicting_episodes(self):
        online_xy = torch.tensor(
            [[[0.3, 0.5], [0.7, 0.5]], [[0.3, 0.5], [0.7, 0.5]]],
            requires_grad=True,
        )
        teacher_xy = torch.tensor(
            [[[0.2, 0.5], [0.8, 0.5]], [[0.2, 0.5], [0.8, 0.5]]]
        )
        output = {
            "online_region_xy": online_xy,
            "geometry_teacher_prompt_conflicts": torch.tensor([[False, False], [True, False]]),
            "geometry_teacher_region_xy": teacher_xy,
            "geometry_teacher_positive_slot_indices": torch.tensor([[0], [0]]),
            "geometry_teacher_negative_slot_indices": torch.tensor([[1], [1]]),
        }
        batch = {
            "positive_xy": torch.tensor([[[0.25, 0.5]], [[0.25, 0.5]]]),
            "negative_xy": torch.tensor([[[0.75, 0.5]], [[0.75, 0.5]]]),
            "positive_mask": torch.ones(2, 1, dtype=torch.bool),
            "negative_mask": torch.ones(2, 1, dtype=torch.bool),
        }
        loss, episodes, prompts = prompt_safe_geometry_anchor_loss(output, batch)
        self.assertEqual(int(episodes), 1); self.assertEqual(int(prompts), 2)
        self.assertAlmostEqual(float(loss.detach()), 0.1, places=5)
        loss.backward()
        self.assertGreater(float(online_xy.grad[0].abs().sum()), 0)
        self.assertEqual(float(online_xy.grad[1].abs().sum()), 0.0)

    def test_geometry_anchor_requires_explicit_teacher_output(self):
        output = {
            "online_region_xy": torch.zeros(1, 1, 2),
        }
        with self.assertRaisesRegex(ValueError, "frozen teacher outputs"):
            prompt_safe_geometry_anchor_loss(
                output, {}
            )

    def test_online_region_loss_supports_single_class_episodes(self):
        logits = torch.tensor([[-1.0, 0.5, -0.2], [1.0, -0.5, 0.2]], requires_grad=True)
        output = {
            "logits": logits,
            "similarity_difference": torch.zeros_like(logits),
        }
        target = torch.tensor([[0, 0, 255], [1, 0, 255]])
        loss, parts = prompt_region_loss(
            output, target, torch.zeros_like(target, dtype=torch.bool), 255,
            dice_weight=1.0, ranking_weight=0.1, ranking_margin=0.2,
            allow_single_class_episodes=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(parts["valid_episodes"]), 2)
        self.assertEqual(int(parts["positive_evaluable_episodes"]), 1)
        self.assertEqual(int(parts["negative_evaluable_episodes"]), 2)
        self.assertEqual(int(parts["ranking_evaluable_episodes"]), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_pixel_loss_reaches_phase2_assignment_and_embedding(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
        )
        batch = make_batch(); output = model(batch)
        self.assertEqual(output["pixel_probability"].shape, (2, 8, 8))
        self.assertEqual(output["online_binary_target"].shape, (2, 4))
        self.assertEqual(output["online_all_prompted_regions"].shape, (2, 4))
        loss, parts = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1, "region_aux": 0.25,
             "region_ranking": 0, "assignment_balance": 0.01, "assignment_entropy": 0.001,
             "assignment_compactness": 0.01},
            0.2,
        )
        self.assertTrue(torch.isfinite(loss)); self.assertIn("pixel_boundary_loss", parts)
        self.assertIn("prompt_separation_loss", parts)
        self.assertIn("teacher_logit_loss", parts)
        self.assertIn("prompt_conflict_episodes", parts)
        self.assertEqual(int(parts["prompt_episodes"]), len(batch["image"]))
        loss.backward()
        self.assertGreater(float(phase2.assignment.weight.grad.abs().sum()), 0)
        self.assertGreater(float(phase2.embedding.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in cell.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in prompt.parameters()))

    def test_decoder_only_freezes_phase2_and_backpropagates_decoder(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
        )
        batch = make_batch(); output = model(batch)
        loss, _ = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1,
             "region_aux": 0.25, "region_ranking": 0,
             "assignment_balance": 0, "assignment_entropy": 0,
             "assignment_compactness": 0},
            0.2,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in phase2.parameters()))
        decoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.parameters() if parameter.grad is not None
        )
        self.assertGreater(decoder_grad, 0.0)

    def test_j3_cell_finetune_freezes_geometry_and_backpropagates_cell(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=1, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
            train_cell=True, train_prompt=False,
        )
        batch = make_batch(); output = model(batch)
        loss, _ = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1,
             "region_aux": 0.25, "region_ranking": 0,
             "assignment_balance": 0, "assignment_entropy": 0,
             "assignment_compactness": 0},
            0.2,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in phase2.parameters()))
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in cell.parameters() if parameter.grad is not None
        ), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in prompt.parameters()))

    def test_j3_partial_prompt_finetune_uses_independent_frozen_teacher(self):
        phase2 = TinyPhase2(); cell = CellToRegionAttention(16, 4)
        prompt = PromptRegionModel(dim=16, heads=4, set_layers=2, dropout=0.0)
        model = JointPromptMaskModel(
            phase2, cell, prompt, region_dim=16, graph_heads=4, graph_layers=1,
            graph_neighbours=2, graph_dropout=0.0,
            train_phase2_embedding=False, train_phase2_assignment=False,
            train_cell=False, train_prompt=True,
        )
        teacher = FrozenPromptTeacher(model.decoder.prompt_model)
        trainable = {
            name for name, parameter in model.decoder.prompt_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith(("matcher.", "task_projection.", "set_pool.encoder.layers.1."))
            for name in trainable
        ))
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        batch = make_batch(); output = model(batch, prompt_teacher=teacher)
        loss, _ = joint_pixel_loss(
            output, batch, 255,
            {"pixel_bce": 1, "pixel_dice": 1, "pixel_boundary": 0.1,
             "region_aux": 0.25, "region_ranking": 0,
             "assignment_balance": 0, "assignment_entropy": 0,
             "assignment_compactness": 0, "teacher_logit": 0.25,
             "teacher_task": 0.05},
            0.2,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in phase2.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in cell.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.prompt_model.parameters()
            if parameter.grad is not None
        ), 0.0)


if __name__ == "__main__":
    unittest.main()
