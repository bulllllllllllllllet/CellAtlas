"""Hard-gated Pareto checkpoint selection for joint pixel training."""
from __future__ import annotations

import math
from typing import Any


MAXIMIZE = (
    "pixel_macro_dice",
    "pixel_micro_dice",
    "region_macro_dice",
    "unprompted_macro_dice",
    "boundary_f1",
)
MINIMIZE = ("prompt_conflict_episode_rate",)
METRICS = (*MAXIMIZE, *MINIMIZE)


def _finite_metrics(row: dict[str, Any]) -> bool:
    return all(metric in row and math.isfinite(float(row[metric])) for metric in METRICS)


def evaluate_hard_gates(row: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    required = {"pixel_macro_dice_min", "pixel_micro_dice_min"}
    missing = required - set(gates)
    if missing:
        raise KeyError(f"checkpoint hard gates missing: {sorted(missing)}")
    checks = {
        "finite_metrics": _finite_metrics(row),
        "pixel_macro_dice_min": float(row.get("pixel_macro_dice", float("nan"))) >= float(gates["pixel_macro_dice_min"]),
        "pixel_micro_dice_min": float(row.get("pixel_micro_dice", float("nan"))) >= float(gates["pixel_micro_dice_min"]),
    }
    violations = [name for name, passed in checks.items() if not passed]
    return {"eligible": not violations, "checks": checks, "violations": violations}


def evaluate_soft_targets(row: dict[str, Any], targets: dict[str, float]) -> dict[str, Any]:
    required = {
        "region_macro_dice_min",
        "unprompted_macro_dice_min",
        "boundary_f1_min",
        "prompt_conflict_episode_rate_max",
    }
    missing = required - set(targets)
    if missing:
        raise KeyError(f"checkpoint soft targets missing: {sorted(missing)}")
    checks = {
        "region_macro_dice_min": float(row.get("region_macro_dice", float("nan"))) >= float(targets["region_macro_dice_min"]),
        "unprompted_macro_dice_min": float(row.get("unprompted_macro_dice", float("nan"))) >= float(targets["unprompted_macro_dice_min"]),
        "boundary_f1_min": float(row.get("boundary_f1", float("nan"))) >= float(targets["boundary_f1_min"]),
        "prompt_conflict_episode_rate_max": float(row.get("prompt_conflict_episode_rate", float("nan"))) <= float(targets["prompt_conflict_episode_rate_max"]),
    }
    return {"checks": checks, "warnings": [name for name, passed in checks.items() if not passed]}


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = all(float(left[name]) >= float(right[name]) for name in MAXIMIZE)
    no_worse &= all(float(left[name]) <= float(right[name]) for name in MINIMIZE)
    strictly_better = any(float(left[name]) > float(right[name]) for name in MAXIMIZE)
    strictly_better |= any(float(left[name]) < float(right[name]) for name in MINIMIZE)
    return no_worse and strictly_better


def build_pareto_report(
    history: list[dict[str, Any]],
    gates: dict[str, float],
    soft_targets: dict[str, float],
    dice_reference: dict[str, float],
    noninferiority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_reference = {"pixel_macro_dice", "pixel_micro_dice"}
    missing_reference = required_reference - set(dice_reference)
    if missing_reference:
        raise KeyError(f"checkpoint Dice reference missing: {sorted(missing_reference)}")
    evaluations = []
    eligible = []
    for row in history:
        gate = evaluate_hard_gates(row, gates)
        soft = evaluate_soft_targets(row, soft_targets)
        summary = {
            "epoch": int(row["epoch"]),
            "path": row.get("checkpoint_path"),
            **{name: float(row[name]) for name in METRICS if name in row},
            **gate,
            "soft_checks": soft["checks"],
            "soft_warnings": soft["warnings"],
        }
        summary["pixel_macro_dice_gain"] = summary["pixel_macro_dice"] - float(dice_reference["pixel_macro_dice"])
        summary["pixel_micro_dice_gain"] = summary["pixel_micro_dice"] - float(dice_reference["pixel_micro_dice"])
        summary["best_pixel_dice_gain"] = max(
            summary["pixel_macro_dice_gain"], summary["pixel_micro_dice_gain"]
        )
        summary["any_pixel_dice_improved"] = summary["best_pixel_dice_gain"] > 0
        if noninferiority is not None:
            macro_margin = float(noninferiority["pixel_macro_dice_margin"])
            micro_margin = float(noninferiority["pixel_micro_dice_margin"])
            require_improvement = bool(
                noninferiority.get("require_any_pixel_dice_improved", False)
            )
            checks = {
                "pixel_macro_dice_noninferior": summary["pixel_macro_dice_gain"] >= -macro_margin,
                "pixel_micro_dice_noninferior": summary["pixel_micro_dice_gain"] >= -micro_margin,
                "any_pixel_dice_improved": (
                    summary["any_pixel_dice_improved"] if require_improvement else True
                ),
            }
            summary["noninferiority_checks"] = checks
            summary["checks"].update(checks)
            summary["violations"].extend(name for name, passed in checks.items() if not passed)
            summary["eligible"] = not summary["violations"]
        evaluations.append(summary)
        if summary["eligible"]:
            if not summary["path"]:
                raise RuntimeError(f"eligible epoch {summary['epoch']} has no checkpoint path")
            eligible.append(summary)
    frontier = [
        candidate for candidate in eligible
        if not any(dominates(other, candidate) for other in eligible if other is not candidate)
    ]
    frontier.sort(key=lambda item: item["epoch"])
    recommended = max(
        frontier,
        key=lambda item: (
            item["best_pixel_dice_gain"],
            item["pixel_macro_dice_gain"], item["pixel_micro_dice_gain"],
            item["region_macro_dice"], item["unprompted_macro_dice"],
            item["boundary_f1"], -item["prompt_conflict_episode_rate"],
        ),
    ) if frontier else None
    return {
        "status": "pareto_frontier" if frontier else "no_eligible_checkpoint",
        "selection": "hard_gates_then_pareto_non_dominance",
        "objectives": {"maximize": list(MAXIMIZE), "minimize": list(MINIMIZE)},
        "hard_gates": {name: float(value) for name, value in gates.items()},
        "soft_targets": {name: float(value) for name, value in soft_targets.items()},
        "dice_reference": {name: float(value) for name, value in dice_reference.items()},
        "noninferiority": (
            {
                "pixel_macro_dice_margin": float(noninferiority["pixel_macro_dice_margin"]),
                "pixel_micro_dice_margin": float(noninferiority["pixel_micro_dice_margin"]),
                "require_any_pixel_dice_improved": bool(
                    noninferiority.get("require_any_pixel_dice_improved", False)
                ),
            }
            if noninferiority is not None else None
        ),
        "recommendation_priority": [
            "best_pixel_dice_gain", "pixel_macro_dice_gain", "pixel_micro_dice_gain", "region_macro_dice",
            "unprompted_macro_dice", "boundary_f1", "prompt_conflict_episode_rate_ascending",
        ],
        "eligible_epochs": [item["epoch"] for item in eligible],
        "frontier": frontier,
        "recommended": recommended,
        "evaluations": evaluations,
    }


def best_checkpoint_pointer(report: dict[str, Any]) -> dict[str, Any]:
    recommended = report["recommended"]
    if recommended is None:
        return {
            "status": "no_eligible_checkpoint",
            "selection": "pixel_dice_hard_gates_then_pareto_dice_priority",
            "path": None,
            "epoch": None,
        }
    return {
        "status": "eligible_checkpoint",
        "selection": "pixel_dice_hard_gates_then_pareto_dice_priority",
        "path": recommended["path"],
        "epoch": recommended["epoch"],
        **{name: recommended[name] for name in METRICS},
        "pixel_macro_dice_gain": recommended["pixel_macro_dice_gain"],
        "pixel_micro_dice_gain": recommended["pixel_micro_dice_gain"],
        "best_pixel_dice_gain": recommended["best_pixel_dice_gain"],
        "any_pixel_dice_improved": recommended["any_pixel_dice_improved"],
        "soft_warnings": recommended["soft_warnings"],
    }
