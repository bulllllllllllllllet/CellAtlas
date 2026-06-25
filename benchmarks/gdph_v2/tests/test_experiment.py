from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import initialize_experiment, sha256_file
from benchmarks.gdph_v2.select_pilot import choose_pilot_rows
from benchmarks.gdph_v2.select_main import assign_image_folds, choose_main_rows
from benchmarks.gdph_v2.eval_crossval import bootstrap_mean_ci, stratified_subsample_indices


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_initialize_experiment(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("image_id\ncase\n", encoding="utf-8")
    palette = tmp_path / "palette.json"
    palette.write_text('{"classes": []}', encoding="utf-8")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "out"
    config = initialize_experiment(
        output, mapping, palette, checkpoint, checkpoint, checkpoint, tmp_path
    )
    assert config["dataset"]["external_to_xcellformer_training"] is True
    assert (output / "config" / "experiment.json").is_file()
    assert (output / "manifests" / "mapping.csv").is_file()


def test_choose_pilot_rows_assigns_distinct_parts(tmp_path: Path) -> None:
    selected = []
    prepared = []
    for part_index in range(5):
        for target_index in range(5):
            image_id = f"p{part_index}_t{target_index}"
            selected.append({"image_id": image_id, "part": f"part{part_index + 1}"})
            mask = np.full((16, 16), 11, dtype=np.uint8)
            target_class = (0, 1, 7, 8, 10)[target_index]
            if part_index == target_index:
                mask[:] = target_class
            path = tmp_path / f"{image_id}.npy"
            np.save(path, mask)
            prepared.append({"image_id": image_id, "gt_mask_path": str(path)})

    rows = choose_pilot_rows(selected, prepared, stride=1)
    assert len(rows) == 5
    assert len({row["part"] for row in rows}) == 5
    assert len({row["image_id"] for row in rows}) == 5


def test_choose_main_rows_keeps_pilots_and_balances_folds() -> None:
    rows = []
    profiles = {}
    pilots = set()
    for part in range(5):
        for index in range(8):
            image_id = f"p{part}_i{index}"
            rows.append({"image_id": image_id, "part": f"part{part + 1}"})
            profile = np.zeros(12)
            profile[(part + index) % 12] = 1
            profiles[image_id] = profile
        pilots.add(f"p{part}_i0")
    selected = choose_main_rows(rows, profiles, pilots, per_part=4)
    assert len(selected) == 20
    assert pilots <= {row["image_id"] for row in selected}
    folded = assign_image_folds(selected)
    counts = np.bincount([int(row["fold"]) for row in folded], minlength=5)
    assert counts.tolist() == [4, 4, 4, 4, 4]


def test_bootstrap_mean_ci_contains_mean() -> None:
    mean, low, high = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=1000)
    assert low <= mean <= high


def test_stratified_subsample_is_deterministic_and_preserves_classes() -> None:
    labels = np.asarray([0] * 80 + [1] * 15 + [2] * 5)
    first = stratified_subsample_indices(labels, max_samples=20, seed=7)
    second = stratified_subsample_indices(labels, max_samples=20, seed=7)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {0, 1, 2}
