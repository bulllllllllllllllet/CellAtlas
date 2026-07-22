#!/usr/bin/env python3
"""Seal a one-shot test summary against an immutable pre-test freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--test-stress-set", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"sealed test result already exists: {args.output}")
    for path in (args.freeze_manifest, args.test_summary, args.test_stress_set, args.test_log):
        if not path.is_file():
            raise FileNotFoundError(path)

    frozen = json.loads(args.freeze_manifest.read_text())
    summary = json.loads(args.test_summary.read_text())
    if frozen.get("status") != "frozen_pre_test" or frozen.get("test_evaluated") is not False:
        raise RuntimeError("invalid pre-test freeze manifest")
    if summary.get("split") != "test" or summary.get("test_used") is not True:
        raise RuntimeError("result is not an explicit test evaluation")
    if int(summary.get("episode_count", 0)) != 4000:
        raise RuntimeError("final test must contain exactly 4000 episodes")
    if list(map(float, summary.get("pixel_thresholds", []))) != [float(frozen["candidate"]["pixel_threshold"])]:
        raise RuntimeError("test threshold differs from frozen threshold")
    if Path(summary["inputs"]["joint_checkpoint"]).resolve() != Path(frozen["candidate"]["checkpoint"]).resolve():
        raise RuntimeError("test checkpoint differs from frozen checkpoint")
    for model_name in ("baseline", "joint"):
        load = summary[f"{model_name}_load"]["joint"]
        if load["missing"] or load["unexpected"]:
            raise RuntimeError(f"{model_name} checkpoint was not loaded strictly")
    if "EXIT_CODE=0" not in args.test_log.read_text():
        raise RuntimeError("test log has no successful exit code")

    baseline = summary["models"]["baseline"]["pixel"]
    joint = summary["models"]["joint"]["pixel"]
    gates = frozen["selection"]["hard_gates"]
    checks = {
        "pixel_macro_dice_min": float(joint["macro_dice"]) >= float(gates["pixel_macro_dice_min"]),
        "pixel_micro_dice_min": float(joint["micro_dice"]) >= float(gates["pixel_micro_dice_min"]),
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "selection_frozen_before_test": True,
        "no_post_test_selection": True,
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "test_summary": str(args.test_summary.resolve()),
        "test_summary_sha256": sha256(args.test_summary),
        "test_stress_set": str(args.test_stress_set.resolve()),
        "test_stress_set_sha256": sha256(args.test_stress_set),
        "test_log": str(args.test_log.resolve()),
        "test_log_sha256": sha256(args.test_log),
        "episode_count": int(summary["episode_count"]),
        "pixel_threshold": float(frozen["candidate"]["pixel_threshold"]),
        "pixel_macro_dice": float(joint["macro_dice"]),
        "pixel_micro_dice": float(joint["micro_dice"]),
        "baseline_pixel_macro_dice": float(baseline["macro_dice"]),
        "baseline_pixel_micro_dice": float(baseline["micro_dice"]),
        "pixel_macro_gain": float(joint["macro_dice"]) - float(baseline["macro_dice"]),
        "pixel_micro_gain": float(joint["micro_dice"]) - float(baseline["micro_dice"]),
        "hard_gates": gates,
        "checks": checks,
        "prompt_conflict_episodes": int(summary["joint_prompt_conflict_episodes"]),
        "prompt_conflict_episode_rate": float(summary["joint_prompt_conflict_episode_rate"]),
        "inference_conflict_action": frozen["prompt_conflict_policy"]["inference_action"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
