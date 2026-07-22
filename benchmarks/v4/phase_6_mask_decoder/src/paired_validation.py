"""Paired bootstrap statistics for fixed-episode checkpoint comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import dice_from_counts


COUNT_NAMES = ("tp", "fp", "fn")


def _arrays(frame: pd.DataFrame, model: str) -> dict[str, np.ndarray]:
    return {
        name: frame[f"{model}_pixel_{name}"].to_numpy(dtype=np.float64)
        for name in COUNT_NAMES
    }


def paired_point_estimates(
    frame: pd.DataFrame, baseline: str = "baseline", candidate: str = "joint"
) -> dict[str, float | int]:
    base = _arrays(frame, baseline); joint = _arrays(frame, candidate)
    base_episode = dice_from_counts(base["tp"], base["fp"], base["fn"])
    joint_episode = dice_from_counts(joint["tp"], joint["fp"], joint["fn"])
    paired = np.isfinite(base_episode) & np.isfinite(joint_episode)
    if not paired.any():
        raise ValueError("no jointly evaluable pixel Dice episodes")
    base_micro = float(dice_from_counts(*(base[name].sum() for name in COUNT_NAMES)))
    joint_micro = float(dice_from_counts(*(joint[name].sum() for name in COUNT_NAMES)))
    return {
        "episodes": int(len(frame)),
        "paired_evaluable_episodes": int(paired.sum()),
        "baseline_macro_dice": float(base_episode[paired].mean()),
        "candidate_macro_dice": float(joint_episode[paired].mean()),
        "macro_dice_difference": float((joint_episode[paired] - base_episode[paired]).mean()),
        "baseline_micro_dice": base_micro,
        "candidate_micro_dice": joint_micro,
        "micro_dice_difference": joint_micro - base_micro,
    }


def paired_episode_bootstrap(
    frame: pd.DataFrame,
    samples: int = 10_000,
    seed: int = 20260722,
    batch_size: int = 128,
    noninferiority_margin: float = 0.001,
) -> dict:
    if len(frame) < 2 or samples < 1 or batch_size < 1:
        raise ValueError("bootstrap requires at least two episodes and positive samples/batch_size")
    base = _arrays(frame, "baseline"); joint = _arrays(frame, "joint")
    base_episode = dice_from_counts(base["tp"], base["fp"], base["fn"])
    joint_episode = dice_from_counts(joint["tp"], joint["fp"], joint["fn"])
    episode_difference = joint_episode - base_episode
    if not np.isfinite(episode_difference).all():
        raise ValueError("paired bootstrap requires finite pixel Dice for every episode")
    rng = np.random.default_rng(int(seed)); n = len(frame)
    macro_draws = np.empty(samples, dtype=np.float64)
    micro_draws = np.empty(samples, dtype=np.float64)
    written = 0
    while written < samples:
        count = min(int(batch_size), samples - written)
        indices = rng.integers(0, n, size=(count, n))
        macro_draws[written:written + count] = episode_difference[indices].mean(axis=1)
        totals = {
            (model, name): values[name][indices].sum(axis=1)
            for model, values in (("baseline", base), ("joint", joint))
            for name in COUNT_NAMES
        }
        base_micro = dice_from_counts(*(totals[("baseline", name)] for name in COUNT_NAMES))
        joint_micro = dice_from_counts(*(totals[("joint", name)] for name in COUNT_NAMES))
        micro_draws[written:written + count] = joint_micro - base_micro
        written += count

    point = paired_point_estimates(frame)
    margin = float(noninferiority_margin)

    def summarize(draws: np.ndarray, point_value: float) -> dict:
        low, high = np.quantile(draws, (0.025, 0.975))
        return {
            "difference": float(point_value),
            "ci_95_percentile": [float(low), float(high)],
            "probability_improved": float(np.mean(draws > 0.0)),
            "probability_noninferior": float(np.mean(draws >= -margin)),
            "ci_demonstrates_improvement": bool(low > 0.0),
            "ci_demonstrates_noninferiority": bool(low >= -margin),
        }

    macro = summarize(macro_draws, float(point["macro_dice_difference"]))
    micro = summarize(micro_draws, float(point["micro_dice_difference"]))
    return {
        "method": "paired_episode_percentile_bootstrap",
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "noninferiority_margin": margin,
        "point_estimates": point,
        "macro_dice": macro,
        "micro_dice": micro,
        "point_rule_passed": bool(
            point["macro_dice_difference"] >= -margin
            and point["micro_dice_difference"] >= -margin
            and max(point["macro_dice_difference"], point["micro_dice_difference"]) > 0.0
        ),
        "ci_noninferiority_passed": bool(
            macro["ci_demonstrates_noninferiority"]
            and micro["ci_demonstrates_noninferiority"]
        ),
    }
