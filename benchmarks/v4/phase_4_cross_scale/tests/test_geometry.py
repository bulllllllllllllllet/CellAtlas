#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from benchmarks.v4.phase_4_cross_scale.src.geometry import (
    assignment_centroids_areas,
    edge_invariants,
    parent_child_edges,
    scale_level0_box,
)


class GeometryTests(unittest.TestCase):
    def test_scale_boxes_nested(self):
        row = {
            "x_level0": 1000,
            "y_level0": 2000,
            "width_level0": 1024,
            "height_level0": 1024,
            "x_5x_level0": 488,
            "y_5x_level0": 1488,
            "width_5x_level0": 2048,
            "height_5x_level0": 2048,
            "x_2p5x_level0": -536,
            "y_2p5x_level0": 464,
            "width_2p5x_level0": 4096,
            "height_2p5x_level0": 4096,
        }
        # fix invalid example for nesting using correct centered boxes
        row = {
            "x_level0": 7168,
            "y_level0": 2048,
            "width_level0": 1024,
            "height_level0": 1024,
            "x_5x_level0": 6656,
            "y_5x_level0": 1536,
            "width_5x_level0": 2048,
            "height_5x_level0": 2048,
            "x_2p5x_level0": 5632,
            "y_2p5x_level0": 512,
            "width_2p5x_level0": 4096,
            "height_2p5x_level0": 4096,
        }
        f = scale_level0_box(row, "10x")
        m = scale_level0_box(row, "5x")
        c = scale_level0_box(row, "2p5x")
        self.assertEqual(f, (7168, 2048, 1024, 1024))
        self.assertTrue(f[0] >= m[0] and f[1] >= m[1])
        self.assertTrue(m[0] >= c[0] and m[1] >= c[1])

    def test_parent_child_weight_sum(self):
        # child has one active blob in center; parent has matching soft mass
        child = np.zeros((4, 8, 8), dtype=np.float32)
        child[0, 3:5, 3:5] = 1.0
        child = child / child.sum(0, keepdims=True).clip(min=1e-6)
        parent = np.zeros((4, 8, 8), dtype=np.float32)
        parent[1, :, :] = 1.0
        parent = parent / parent.sum(0, keepdims=True)
        child_box = (100, 100, 100, 100)
        parent_box = (50, 50, 200, 200)
        edges = parent_child_edges(child, child_box, parent, parent_box, top_k=2)
        inv = edge_invariants(edges)
        self.assertTrue(inv["passed"], inv)
        self.assertEqual(int(edges["edge_index"][0, 0]), 1)

    def test_centroids_inside_box(self):
        assignment = np.zeros((4, 16, 16), dtype=np.float32)
        assignment[0, 2:4, 2:4] = 1.0
        assignment = assignment / np.maximum(assignment.sum(0, keepdims=True), 1e-6)
        geom = assignment_centroids_areas(assignment, 1000, 2000, 512, 512)
        self.assertTrue(geom["active"][0])
        self.assertTrue(1000 <= geom["centroid_x"][0] <= 1000 + 512)
        self.assertTrue(2000 <= geom["centroid_y"][0] <= 2000 + 512)


if __name__ == "__main__":
    unittest.main()
