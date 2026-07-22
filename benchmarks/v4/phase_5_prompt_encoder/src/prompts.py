"""Deterministic target-vs-rest prompt episode sampling over fine regions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROMPT_SIZE_SPECS = {
    "point": {"min_slots": 1, "max_slots": 1, "min_fraction": 0.003, "max_fraction": 0.035, "negative_slots": 1},
    "small": {"min_slots": 2, "max_slots": 3, "min_fraction": 0.03, "max_fraction": 0.06, "negative_slots": 2},
    "large": {"min_slots": 4, "max_slots": 8, "min_fraction": 0.08, "max_fraction": 0.15, "negative_slots": 3},
}


@dataclass(frozen=True)
class PromptEpisode:
    """A prompt episode after point/region prompts have been mapped to slots."""

    target_class: int
    positive_slots: np.ndarray
    negative_slots: np.ndarray
    positive_xy: np.ndarray
    negative_xy: np.ndarray
    binary_region_target: np.ndarray

    def as_dict(self) -> dict:
        return {
            "target_class": int(self.target_class),
            "positive_slots": self.positive_slots.astype(int).tolist(),
            "negative_slots": self.negative_slots.astype(int).tolist(),
            "positive_xy": self.positive_xy.astype(float).tolist(),
            "negative_xy": self.negative_xy.astype(float).tolist(),
            "binary_region_target": self.binary_region_target.astype(int).tolist(),
        }


def hard_region_adjacency(hard: np.ndarray, num_slots: int) -> np.ndarray:
    """Return exact 4-neighbour region adjacency from a hard assignment map."""
    hard = np.asarray(hard)
    if hard.ndim != 2:
        raise ValueError(f"hard assignment must be [H,W], got {hard.shape}")
    if hard.size and (int(hard.min()) < 0 or int(hard.max()) >= num_slots):
        raise ValueError("hard assignment contains an out-of-range slot")
    adjacency = np.zeros((num_slots, num_slots), dtype=bool)
    for one, two in ((hard[:, :-1], hard[:, 1:]), (hard[:-1], hard[1:])):
        changed = one != two
        left = one[changed].astype(np.int64)
        right = two[changed].astype(np.int64)
        adjacency[left, right] = True
        adjacency[right, left] = True
    np.fill_diagonal(adjacency, False)
    return adjacency


def centroid_knn_adjacency(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    active: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Approximate fine-region adjacency from cached physical centroids."""
    x = np.asarray(centroid_x, dtype=np.float64)
    y = np.asarray(centroid_y, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if x.ndim != 1 or y.shape != x.shape or active.shape != x.shape:
        raise ValueError("centroid/active shape mismatch")
    if k < 1:
        raise ValueError("k must be positive")
    adjacency = np.zeros((len(x), len(x)), dtype=bool)
    ids = np.flatnonzero(active)
    if ids.size <= 1:
        return adjacency
    points = np.stack([x[ids], y[ids]], axis=1)
    distance = ((points[:, None] - points[None]) ** 2).sum(2)
    np.fill_diagonal(distance, np.inf)
    neighbours = min(k, ids.size - 1)
    for local, slot in enumerate(ids):
        nearest = ids[np.argpartition(distance[local], neighbours - 1)[:neighbours]]
        adjacency[slot, nearest] = True
        adjacency[nearest, slot] = True
    np.fill_diagonal(adjacency, False)
    return adjacency


def _is_connected(slots: np.ndarray, adjacency: np.ndarray) -> bool:
    if slots.size <= 1:
        return True
    allowed = set(map(int, slots))
    visited = {int(slots[0])}
    frontier = [int(slots[0])]
    while frontier:
        current = frontier.pop()
        for neighbour in np.flatnonzero(adjacency[current]):
            neighbour = int(neighbour)
            if neighbour in allowed and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == allowed


def sample_connected_region_set(
    adjacency: np.ndarray,
    eligible: np.ndarray,
    region_pixels: np.ndarray,
    rng: np.random.Generator,
    *,
    min_slots: int,
    max_slots: int,
    min_fraction: float,
    max_fraction: float,
    patch_pixels: int,
) -> np.ndarray:
    """Grow one connected eligible slot set within explicit size bounds."""
    adjacency = np.asarray(adjacency, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    region_pixels = np.asarray(region_pixels, dtype=np.int64)
    k = eligible.size
    if adjacency.shape != (k, k) or region_pixels.shape != (k,):
        raise ValueError("adjacency/eligible/region_pixels shape mismatch")
    if not (1 <= min_slots <= max_slots <= k):
        raise ValueError("invalid slot bounds")
    if not (0 < min_fraction <= max_fraction <= 1):
        raise ValueError("invalid area fraction bounds")
    if patch_pixels <= 0:
        raise ValueError("patch_pixels must be positive")

    minimum_pixels = min_fraction * patch_pixels
    maximum_pixels = max_fraction * patch_pixels
    seeds = np.flatnonzero(eligible)
    rng.shuffle(seeds)
    valid_sets: list[np.ndarray] = []
    for seed in seeds:
        selected = [int(seed)]
        while len(selected) < max_slots:
            area = int(region_pixels[selected].sum())
            if len(selected) >= min_slots and minimum_pixels <= area <= maximum_pixels:
                valid_sets.append(np.asarray(sorted(selected), dtype=np.int64))
                break
            neighbours = np.flatnonzero(adjacency[selected].any(0) & eligible)
            neighbours = np.setdiff1d(neighbours, np.asarray(selected), assume_unique=False)
            if neighbours.size == 0:
                break
            # Prefer an addition that approaches the target interval without
            # exceeding its upper bound; randomise ties deterministically.
            rng.shuffle(neighbours)
            candidate_area = area + region_pixels[neighbours]
            within = neighbours[candidate_area <= maximum_pixels]
            if within.size:
                distance = np.abs(
                    (area + region_pixels[within]) - 0.5 * (minimum_pixels + maximum_pixels)
                )
                next_slot = int(within[int(distance.argmin())])
            else:
                break
            selected.append(next_slot)
        else:
            area = int(region_pixels[selected].sum())
            if len(selected) >= min_slots and minimum_pixels <= area <= maximum_pixels:
                valid_sets.append(np.asarray(sorted(selected), dtype=np.int64))
    if not valid_sets:
        raise ValueError(
            f"no connected prompt set satisfies slots={min_slots}-{max_slots}, "
            f"fraction={min_fraction:.3f}-{max_fraction:.3f}"
        )
    chosen = valid_sets[int(rng.integers(0, len(valid_sets)))]
    if not _is_connected(chosen, adjacency):
        raise RuntimeError("sampled prompt set is not connected")
    return chosen


def dominant_parent_slots(
    child_slots: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    """Map selected child slots to their dominant valid parent slots."""
    child_slots = np.asarray(child_slots, dtype=np.int64)
    edge_index = np.asarray(edge_index)
    edge_weight = np.asarray(edge_weight)
    if edge_index.shape != edge_weight.shape or edge_index.ndim != 2:
        raise ValueError("edge index/weight shape mismatch")
    if child_slots.size == 0:
        return np.empty(0, dtype=np.int64)
    parents = []
    for child in child_slots:
        indices = edge_index[child]
        weights = edge_weight[child]
        valid = (indices >= 0) & (weights > 0)
        if valid.any():
            local = np.flatnonzero(valid)
            parents.append(int(indices[local[int(weights[valid].argmax())]]))
    return np.unique(np.asarray(parents, dtype=np.int64))


def _normalised_xy(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    box_level0: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, width, height = box_level0
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid patch box: {box_level0}")
    xy = np.stack(
        [
            (centroid_x.astype(np.float64) - x0) / width,
            (centroid_y.astype(np.float64) - y0) / height,
        ],
        axis=1,
    )
    return xy.astype(np.float32)


def sample_region_prompt_episode(
    labels: np.ndarray,
    active: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    box_level0: tuple[int, int, int, int],
    rng: np.random.Generator,
    *,
    target_class: int | None = None,
    target_class_ids: tuple[int, ...] | None = None,
    min_positive: int = 1,
    max_positive: int = 4,
    min_negative: int = 0,
    max_negative: int = 4,
    ignore_index: int = 255,
    eligible_positive: np.ndarray | None = None,
    eligible_negative: np.ndarray | None = None,
    hard_negative_fraction: float = 0.5,
) -> PromptEpisode:
    """Sample one deterministic visual-prompt episode.

    Prompt coordinates are normalised cached centroids. During interactive
    inference a raw point is first mapped to the same hard fine-region slot;
    the prompt encoder consumes the selected slot token and coordinates.
    """
    labels = np.asarray(labels)
    active = np.asarray(active, dtype=bool)
    centroid_x = np.asarray(centroid_x)
    centroid_y = np.asarray(centroid_y)
    if labels.ndim != 1:
        raise ValueError(f"labels must be [K], got {labels.shape}")
    k = labels.shape[0]
    for name, value in (
        ("active", active),
        ("centroid_x", centroid_x),
        ("centroid_y", centroid_y),
    ):
        if value.shape != (k,):
            raise ValueError(f"{name} must have shape {(k,)}, got {value.shape}")
    if not (1 <= min_positive <= max_positive):
        raise ValueError("positive prompt bounds must satisfy 1 <= min <= max")
    if not (0 <= min_negative <= max_negative):
        raise ValueError("negative prompt bounds must satisfy 0 <= min <= max")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0,1]")

    valid = active & (labels != ignore_index)
    if eligible_positive is not None:
        eligible_positive = np.asarray(eligible_positive, dtype=bool)
        if eligible_positive.shape != (k,):
            raise ValueError("eligible_positive shape mismatch")
    else:
        eligible_positive = np.ones(k, dtype=bool)
    if eligible_negative is not None:
        eligible_negative = np.asarray(eligible_negative, dtype=bool)
        if eligible_negative.shape != (k,):
            raise ValueError("eligible_negative shape mismatch")
    else:
        eligible_negative = np.ones(k, dtype=bool)

    present = np.unique(labels[valid & eligible_positive]).astype(int)
    if target_class_ids is not None:
        allowed = np.asarray(target_class_ids, dtype=int)
        present = present[np.isin(present, allowed)]
    if target_class is None:
        if present.size == 0:
            raise ValueError("no eligible target class in patch")
        target_class = int(rng.choice(present))
    elif int(target_class) not in present:
        raise ValueError(f"target class {target_class} has no eligible positive region")
    target_class = int(target_class)

    positive_pool = np.flatnonzero(valid & eligible_positive & (labels == target_class))
    negative_pool = np.flatnonzero(valid & eligible_negative & (labels != target_class))
    if positive_pool.size < min_positive:
        raise ValueError(f"only {positive_pool.size} positive regions, need {min_positive}")
    if negative_pool.size < min_negative:
        raise ValueError(f"only {negative_pool.size} negative regions, need {min_negative}")

    n_positive = int(rng.integers(min_positive, min(max_positive, positive_pool.size) + 1))
    positive_slots = np.sort(rng.choice(positive_pool, size=n_positive, replace=False)).astype(np.int64)
    n_negative = int(rng.integers(min_negative, min(max_negative, negative_pool.size) + 1))

    xy = _normalised_xy(centroid_x, centroid_y, box_level0)
    if n_negative == 0:
        negative_slots = np.empty(0, dtype=np.int64)
    else:
        # Spatially close negatives are useful boundary confounders. Mix them
        # with random negatives without depending on a trained classifier.
        distance = ((xy[negative_pool, None] - xy[positive_slots][None]) ** 2).sum(2).min(1)
        ordered = negative_pool[np.argsort(distance, kind="stable")]
        n_hard = min(n_negative, int(round(n_negative * hard_negative_fraction)))
        chosen_hard = ordered[:n_hard]
        remainder = np.setdiff1d(negative_pool, chosen_hard, assume_unique=True)
        chosen_random = rng.choice(remainder, size=n_negative - n_hard, replace=False)
        negative_slots = np.sort(np.concatenate([chosen_hard, chosen_random])).astype(np.int64)

    binary = np.full(k, ignore_index, dtype=np.int16)
    binary[valid] = (labels[valid] == target_class).astype(np.int16)
    return PromptEpisode(
        target_class=target_class,
        positive_slots=positive_slots,
        negative_slots=negative_slots,
        positive_xy=xy[positive_slots],
        negative_xy=xy[negative_slots],
        binary_region_target=binary,
    )
