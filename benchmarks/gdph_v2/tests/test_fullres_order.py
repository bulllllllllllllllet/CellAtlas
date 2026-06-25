import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("openslide")

from benchmarks.gdph_v2.fullres_inference import (
    _centers_and_labels,
    _deduplicate_cross_tile_records,
)


def test_centers_preserve_cellpose_label_order() -> None:
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[6, 6] = 1
    mask[1, 1] = 2
    labels, centers = _centers_and_labels(mask)
    assert labels.tolist() == [1, 2]
    assert centers.tolist() == [[6.0, 6.0], [1.0, 1.0]]


def _record(index: int, tile_index: int, x: float, y: float, boundary: float) -> dict:
    return {
        "cell_index": index,
        "tile_index": tile_index,
        "local_label": index + 10,
        "x_original": x,
        "y_original": y,
        "x_gt": x / 2,
        "y_gt": y / 2,
        "core_boundary_distance": boundary,
        "polygon_original": [],
    }


def test_cross_tile_dedup_keeps_cell_farther_from_core_boundary() -> None:
    records = [
        _record(0, 0, 100.0, 100.0, 0.5),
        _record(1, 1, 101.0, 100.0, 5.0),
        _record(2, 1, 200.0, 200.0, 1.0),
    ]
    raw = np.arange(9, dtype=np.float32).reshape(3, 3)
    reg = raw + 10
    proj = raw + 20

    deduped, deduped_raw, deduped_reg, deduped_proj, stats = _deduplicate_cross_tile_records(
        records, raw, reg, proj, radius=2.0
    )

    assert [record["local_label"] for record in deduped] == [11, 12]
    assert [record["cell_index"] for record in deduped] == [0, 1]
    assert deduped_raw.tolist() == raw[[1, 2]].tolist()
    assert deduped_reg.tolist() == reg[[1, 2]].tolist()
    assert deduped_proj.tolist() == proj[[1, 2]].tolist()
    assert stats == {
        "cells_before_dedup": 3,
        "cross_tile_near_duplicates_before_dedup": 1,
        "deduplicated_cross_tile_cells": 1,
    }


def test_cross_tile_dedup_does_not_remove_same_tile_neighbors() -> None:
    records = [
        _record(0, 0, 100.0, 100.0, 0.5),
        _record(1, 0, 101.0, 100.0, 5.0),
    ]
    raw = np.arange(4, dtype=np.float32).reshape(2, 2)
    deduped, deduped_raw, _, _, stats = _deduplicate_cross_tile_records(
        records, raw, raw, raw, radius=2.0
    )

    assert len(deduped) == 2
    assert deduped_raw.tolist() == raw.tolist()
    assert stats["deduplicated_cross_tile_cells"] == 0
