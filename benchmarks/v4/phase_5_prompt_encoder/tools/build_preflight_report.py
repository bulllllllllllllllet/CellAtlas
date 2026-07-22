#!/usr/bin/env python3
"""Build the final machine-readable Phase-5 formal-training gate report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eligibility-audit", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--logs", type=Path, nargs="+", required=True)
    p.add_argument("--final-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def main():
    a = parse_args()
    eligibility = json.loads(a.eligibility_audit.read_text())
    metadata = [json.loads((path / "run_metadata.json").read_text()) for path in a.run_dirs]
    payload = torch.load(a.final_checkpoint, map_location="cpu", weights_only=False)
    history = payload["history"]
    final = history[-1]
    required_checkpoint_keys = {"epoch", "model", "optimizer", "scheduler", "scaler", "config", "history"}
    log_exit_codes = ["EXIT_CODE=0" in path.read_text() for path in a.logs]
    class_metrics = final.get("val_by_class", {})
    size_metrics = final.get("val_by_size", {})
    checks = {
        "eligibility_passed": eligibility.get("passed") is True,
        "all_logs_exit_zero": all(log_exit_codes),
        "all_runs_exclude_test": all(item.get("test_used") is False for item in metadata),
        "checkpoint_keys_complete": required_checkpoint_keys <= set(payload),
        "history_epochs_contiguous": [item["epoch"] for item in history] == list(range(int(payload["epoch"]) + 1)),
        "final_epoch_is_3": int(payload["epoch"]) == 3,
        "scheduler_horizon_is_30": int(payload["scheduler"]["T_max"]) == 30,
        "scheduler_advanced_to_4": int(payload["scheduler"]["last_epoch"]) == 4,
        "metrics_all_finite": finite_tree(history),
        "all_12_classes_reported": len(class_metrics) == 12,
        "all_3_sizes_reported": len(size_metrics) == 3,
        "no_class_prediction_collapse": all(0.05 < item["predicted_positive_fraction"] < 0.95 for item in class_metrics.values()),
        "no_size_prediction_collapse": all(0.05 < item["predicted_positive_fraction"] < 0.95 for item in size_metrics.values()),
        "all_class_dice_non_degenerate": all(item["region_dice"] > 0.1 for item in class_metrics.values()),
        "all_size_dice_non_degenerate": all(item["region_dice"] > 0.1 for item in size_metrics.values()),
        "unprompted_recall_non_degenerate": final["unprompted_target_recall"] > 0.1,
        "immutable_checkpoints_present": all(any(path.glob("checkpoint_epoch_*.pth")) for path in a.run_dirs),
        "last_and_best_pointers_present": all((path / "last_checkpoint.json").is_file() and (path / "best_checkpoint.json").is_file() for path in a.run_dirs),
    }
    report = {
        "phase": 5,
        "purpose": "formal training preflight gate",
        "formal_training_started": False,
        "passed": all(checks.values()),
        "checks": checks,
        "eligibility": {
            "rows": eligibility["rows"],
            "completed_rows": eligibility["completed_rows"],
            "unique_eligible_patches": eligibility["unique_eligible_patches"],
            "patches_without_episode": eligibility["patches_without_episode"],
        },
        "resume_chain": [str(path) for path in a.run_dirs],
        "final_checkpoint": str(a.final_checkpoint),
        "history_epochs": [item["epoch"] for item in history],
        "scheduler": payload["scheduler"],
        "final_metrics": final,
        "formal_budget": payload["config"]["training"],
        "test_used": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=False)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
