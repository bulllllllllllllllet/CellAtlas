from __future__ import annotations

from benchmarks.gdph_v2.geometry import generate_core_tiles, map_point_to_gt
from benchmarks.gdph_v2.polygon_labels import label_status, polygon_majority_label
from benchmarks.gdph_v2.generate_queries import box_bounds
from benchmarks.gdph_v2.eval_retrieval import binary_metrics, normalized_median_prototype, ranking_metrics
from benchmarks.gdph_v2.eval_cell_coverage import sampled_oracle_coverage
from benchmarks.gdph_v2.patch_geometry import rectangle_majority_label
from benchmarks.gdph_v2.eval_patch_retrieval import percentile_scores
from benchmarks.gdph_v2.eval_nucleus_detection import one_to_one_point_metrics, stream_instance_centroids

import numpy as np
import pytest


def test_core_tiles_cover_once_and_reads_have_halo() -> None:
    tiles = generate_core_tiles(9000, 7000, tile_size=4096, halo=256)
    assert len(tiles) == 6
    for y in (0, 100, 4095, 4096, 6999):
        for x in (0, 100, 4095, 4096, 8191, 8999):
            assert sum(tile.owns(x, y) for tile in tiles) == 1
    middle = tiles[1]
    assert middle.read_x0 == middle.core_x0 - 256
    assert middle.read_x1 == middle.core_x1 + 256


def test_map_point_to_gt_supports_non_exact_half_scale() -> None:
    x, y = map_point_to_gt(27235, 39025, (27235, 39025), (13617, 19512))
    assert x == 13617
    assert y == 19512


def test_polygon_majority_label_and_purity() -> None:
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[:, 5:] = 1
    label, purity, pixels = polygon_majority_label(
        [[8, 2], [18, 2], [18, 18], [8, 18]],
        (13, 10),
        (20, 20),
        gt,
        num_classes=2,
    )
    assert label == 1
    assert purity > 0.7
    assert pixels > 0
    assert label_status(label, purity) == "valid"


def test_polygon_majority_falls_back_to_centroid() -> None:
    gt = np.full((10, 10), 3, dtype=np.uint8)
    label, purity, pixels = polygon_majority_label([], (10, 10), (20, 20), gt, 4)
    assert (label, purity, pixels) == (3, 1.0, 1)


def test_polygon_purity_counts_ignored_gt_pixels_in_denominator() -> None:
    gt = np.full((10, 10), 255, dtype=np.uint8)
    gt[4:6, 4:6] = 1
    label, purity, pixels = polygon_majority_label(
        [[0, 0], [9, 0], [9, 9], [0, 9]],
        (5, 5),
        (10, 10),
        gt,
        num_classes=2,
    )
    assert label == 1
    assert pixels == 100
    assert purity == 0.04
    assert label_status(label, purity) == "ignore"


def test_box_bounds_stays_inside_slide() -> None:
    assert box_bounds(10, 10, 100, 1000, 800) == (0, 0, 100, 100)
    assert box_bounds(990, 790, 100, 1000, 800) == (900, 700, 1000, 800)


def test_retrieval_prototype_and_metrics() -> None:
    prototype = normalized_median_prototype(np.asarray([[1, 0], [0.9, 0.1]], dtype=np.float32))
    assert np.isclose(np.linalg.norm(prototype), 1)
    metrics = ranking_metrics(np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.8, 0.2, 0.1]), ks=(2,))
    assert metrics["average_precision"] == 1.0
    assert metrics["precision_at_2"] == 1.0


def test_binary_retrieval_metrics_include_iou() -> None:
    metrics = binary_metrics(
        np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.4, 0.6, 0.1]), threshold=0.5
    )
    assert metrics["binary_true_positive"] == 1
    assert metrics["binary_false_positive"] == 1
    assert metrics["binary_iou"] == 1 / 3


def test_sampled_oracle_coverage_reports_unknown_area() -> None:
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[:, 10:] = 1
    cells = np.asarray([[2, 10], [17, 10]], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    small, large = sampled_oracle_coverage(gt, cells, labels, [2, 20], stride=1)
    assert small["coverage"] < 1
    assert large["coverage"] == 1
    assert large["whole_image_oracle_accuracy"] > 0.9
    assert small["per_class"]["0"]["coverage"] < 1
    assert large["per_class"]["1"]["coverage"] == 1


def test_rectangle_majority_label() -> None:
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[:, 5:] = 2
    label, purity, pixels = rectangle_majority_label(gt, (20, 20), (10, 0, 20, 20))
    assert label == 2
    assert purity == 1
    assert pixels == 50


def test_rectangle_purity_counts_ignored_gt_pixels() -> None:
    gt = np.full((10, 10), 255, dtype=np.uint8)
    gt[4:6, 4:6] = 2
    label, purity, pixels = rectangle_majority_label(gt, (10, 10), (0, 0, 10, 10))
    assert label == 2
    assert pixels == 100
    assert purity == 0.04


def test_percentile_scores_preserve_order() -> None:
    scores = percentile_scores(np.asarray([0.5, 0.1, 0.9]))
    assert scores[2] > scores[0] > scores[1]


def test_percentile_scores_average_ties() -> None:
    scores = percentile_scores(np.asarray([0.5, 0.5, 0.1]))
    assert scores[0] == scores[1]


def test_one_to_one_point_metrics_prevents_duplicate_matches() -> None:
    prediction = np.asarray([[0, 0], [0.1, 0.1], [10, 10]], dtype=np.float64)
    truth = np.asarray([[0, 0], [10, 10]], dtype=np.float64)
    metrics = one_to_one_point_metrics(prediction, truth, max_distance=1)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["recall"] == 1


def test_one_to_one_point_metrics_tries_alternate_gt_candidate() -> None:
    prediction = np.asarray([[0, 0], [0.2, 0]], dtype=np.float64)
    truth = np.asarray([[0, 0], [1, 0]], dtype=np.float64)
    metrics = one_to_one_point_metrics(prediction, truth, max_distance=1)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0


def test_one_to_one_point_metrics_maximizes_cardinality_not_greedy_distance() -> None:
    prediction = np.asarray([[0.0, 0], [-0.05, 0]], dtype=np.float64)
    truth = np.asarray([[0.1, 0], [-0.2, 0]], dtype=np.float64)
    # The shortest edge is prediction 0 -> truth 0, but taking it would leave
    # prediction 1 unmatched. Maximum-cardinality matching finds both pairs.
    metrics = one_to_one_point_metrics(prediction, truth, max_distance=0.21)
    assert metrics["true_positive"] == 2


def test_stream_instance_centroids_merges_instances_across_tiff_tiles(tmp_path) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")

    instance = np.zeros((32, 32), dtype=np.int32)
    instance[2:10, 2:10] = 1
    instance[14:20, 14:20] = 2
    path = tmp_path / "instance.tiff"
    tifffile.imwrite(path, instance, tile=(16, 16))
    centroids = stream_instance_centroids(path, tile_size=16)
    assert centroids[:, 0].tolist() == [1, 2]
    assert np.allclose(centroids[:, 1:3], [[5.5, 5.5], [16.5, 16.5]])
