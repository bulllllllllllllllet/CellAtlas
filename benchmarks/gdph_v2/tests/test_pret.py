from __future__ import annotations

import numpy as np

from benchmarks.gdph_v2.pret_compare_scales import row_key, summarize_rows
from benchmarks.gdph_v2.pret_eval_in_context import scores_from_prompt
from benchmarks.gdph_v2.pret_generate_prompts import overlap_segments
from benchmarks.gdph_v2.pret_superpixel_tokens import fallback_grid_segments
from benchmarks.gdph_v2.pret_utils import (
    area_sweep_metrics,
    binary_segmentation_metrics,
    box_original_to_target,
    connected_component_topk_mask,
    majority_label,
    mean_std_threshold,
    normalized_median,
    otsu_score_threshold,
    percentile_threshold,
    precision_at_area,
    prompt_relative_threshold,
    prompt_purity_bin,
    safe_l2_normalize,
    segment_adjacency,
    smooth_scores,
    weighted_concat,
)


def test_box_original_to_target_converts_buffer_scale() -> None:
    box = box_original_to_target((100, 200, 300, 600), (1000, 2000), (1000, 500))
    assert box == (50, 100, 150, 300)


def test_majority_label_uses_valid_pixels_for_purity() -> None:
    values = np.asarray([0, 0, 1, 255])
    label, purity, valid_fraction = majority_label(values, num_classes=2)
    assert label == 0
    assert purity == 2 / 3
    assert valid_fraction == 3 / 4


def test_block_wise_normalization_prevents_large_block_scale_domination() -> None:
    small = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    large = np.asarray([[1000, 0], [0, 1000]], dtype=np.float32)
    token = weighted_concat([(small, 1.0), (large, 1.0)])
    assert np.all(np.isfinite(token))
    assert np.allclose(np.linalg.norm(token[:, :2], axis=1), 1)
    assert np.allclose(np.linalg.norm(token[:, 2:], axis=1), 1)


def test_safe_l2_normalize_handles_zero_cell_segments() -> None:
    normalized = safe_l2_normalize(np.asarray([[0, 0], [3, 4]], dtype=np.float32))
    assert np.all(np.isfinite(normalized))
    assert normalized[0].tolist() == [0.0, 0.0]
    assert np.allclose(normalized[1], [0.6, 0.8])


def test_one_shot_score_does_not_require_prompt_quantile_threshold() -> None:
    tokens = np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
    scores = scores_from_prompt(tokens, positive=[0], negative=[], strategy="negative_bank_max", lambda_neg=0.5)
    assert scores[0] == 1
    assert scores[1] < 1
    assert np.count_nonzero(scores >= 1) == 1
    top_area = precision_at_area(np.asarray([1, 1, 0]), scores, np.ones(3), 2 / 3)
    assert top_area == 1.0


def test_negative_bank_max_penalizes_closest_negative() -> None:
    tokens = np.asarray([[1, 0], [0, 1], [0.1, 0.9]], dtype=np.float32)
    scores = scores_from_prompt(tokens, positive=[0], negative=[1], strategy="negative_bank_max", lambda_neg=0.5)
    assert scores[0] > scores[2]
    assert scores[2] < 0


def test_overlap_segments_uses_segment_fraction_inside_box() -> None:
    segments = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    assert overlap_segments(segments, (0, 0, 2, 2), 0.5) == [0]
    assert overlap_segments(segments, (1, 0, 3, 2), 0.5) == [0, 1]


def test_fallback_grid_segments_writes_ids_inside_mask() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[:3, :3] = True
    segments = fallback_grid_segments(mask, diameter=2)
    assert segments[0, 0] >= 0
    assert segments[4, 4] == -1
    assert len(np.unique(segments[segments >= 0])) > 1


def test_segmentation_metrics_toy_case() -> None:
    metrics = binary_segmentation_metrics(
        np.asarray([1, 1, 0, 0], dtype=bool),
        np.asarray([1, 0, 1, 0], dtype=bool),
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["iou"] == 1 / 3


def test_normalized_median_zero_input_is_finite() -> None:
    proto = normalized_median(np.zeros((2, 3), dtype=np.float32))
    assert proto.tolist() == [0.0, 0.0, 0.0]
    assert np.all(np.isfinite(proto))


def test_area_sweep_best_dice_bounds_fixed_ratios() -> None:
    target = np.asarray([1, 1, 0, 0], dtype=bool)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    areas = np.ones(4)
    metrics = area_sweep_metrics(target, scores, areas, (0.25, 0.5, 0.75))
    assert metrics["top25_area_dice"] <= metrics["best_area_dice"]
    assert metrics["top50_area_dice"] == 1.0
    assert metrics["best_area_ratio"] == 0.5


def test_prompt_purity_bins() -> None:
    assert prompt_purity_bin(0.49) == "<0.5"
    assert prompt_purity_bin(0.5) == "0.5-0.7"
    assert prompt_purity_bin(0.7) == "0.7-0.9"
    assert prompt_purity_bin(0.9) == ">0.9"


def test_neighbor_graph_and_smoothing() -> None:
    segments = np.asarray([[0, 0, 1], [2, 2, 1]], dtype=np.int32)
    neighbors = segment_adjacency(segments, 3)
    assert set(neighbors[0].tolist()) == {1, 2}
    scores = np.asarray([1.0, 0.0, 0.0])
    assert np.allclose(smooth_scores(scores, neighbors, 0.0), scores)
    smoothed = smooth_scores(scores, neighbors, 0.5)
    assert smoothed[0] == 0.5
    assert smoothed[1] > 0


def test_deployable_thresholds_are_finite_and_score_only() -> None:
    scores = np.asarray([0.05, 0.1, 0.2, 0.8, 0.9])
    assert np.isfinite(otsu_score_threshold(scores))
    assert mean_std_threshold(scores, 0.5) > scores.mean()
    assert percentile_threshold(scores, 90) == np.percentile(scores, 90)


def test_prompt_relative_threshold_falls_back_for_one_shot() -> None:
    candidate_scores = np.asarray([0.1, 0.2, 0.3, 0.4])
    positive_scores = np.asarray([1.0])
    threshold = prompt_relative_threshold(candidate_scores, positive_scores, margin=0.1, fallback_percentile=75)
    assert threshold == np.percentile(candidate_scores, 75)


def test_connected_component_topk_keeps_high_score_component() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    areas = np.ones(4)
    neighbors = [
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
    ]
    selected = connected_component_topk_mask(scores, areas, neighbors, seed_fraction=0.75, max_components=1)
    assert selected.tolist() == [True, True, False, False]


def test_scale_compare_uses_query_id_and_shot_key() -> None:
    row_a = {"query_id": "q0", "shot": "1"}
    row_b = {"query_id": "q0", "shot": "3"}
    assert row_key(row_a) != row_key(row_b)


def test_scale_compare_normalized_gain() -> None:
    rows = [{"average_precision": "0.6"}]
    random_rows = [{"average_precision": "0.2"}]
    summary = summarize_rows(rows, random_rows)
    assert summary["average_precision"] == 0.6
    assert summary["random_average_precision"] == 0.2
    assert np.isclose(summary["normalized_map_gain"], 0.5)
