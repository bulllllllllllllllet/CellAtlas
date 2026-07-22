#!/usr/bin/env python3
"""Freeze the validation-selected Phase-6 candidate before test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-stress-set", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--pixel-macro-floor", type=float, default=0.72)
    parser.add_argument("--pixel-micro-floor", type=float, default=0.7987)
    parser.add_argument("--pixel-threshold", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "validation_summary": args.validation_summary,
        "validation_stress_set": args.validation_stress_set,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")

    summary = json.loads(args.validation_summary.read_text())
    if summary.get("split") != "val" or bool(summary.get("test_used")):
        raise RuntimeError("candidate must be frozen from validation-only metrics")
    baseline = summary["models"]["baseline"]["pixel"]
    joint = summary["models"]["joint"]["pixel"]
    macro = float(joint["macro_dice"])
    micro = float(joint["micro_dice"])
    macro_gain = macro - float(baseline["macro_dice"])
    micro_gain = micro - float(baseline["micro_dice"])
    checks = {
        "pixel_macro_floor": macro >= args.pixel_macro_floor,
        "pixel_micro_floor": micro >= args.pixel_micro_floor,
        "any_pixel_dice_improved": max(macro_gain, micro_gain) > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"candidate failed frozen selection checks: {checks}")

    return {
        "schema_version": 1,
        "timestamp": args.timestamp,
        "status": "frozen_pre_test",
        "test_evaluated": False,
        "candidate": {
            "name": "J5_backbone_layer4_epoch1",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "pixel_threshold": args.pixel_threshold,
        },
        "selection": {
            "validation_summary": str(args.validation_summary.resolve()),
            "validation_summary_sha256": sha256(args.validation_summary),
            "validation_stress_set": str(args.validation_stress_set.resolve()),
            "validation_stress_set_sha256": sha256(args.validation_stress_set),
            "episode_count": int(summary["episode_count"]),
            "pixel_macro_dice": macro,
            "pixel_micro_dice": micro,
            "baseline_pixel_macro_dice": float(baseline["macro_dice"]),
            "baseline_pixel_micro_dice": float(baseline["micro_dice"]),
            "pixel_macro_gain": macro_gain,
            "pixel_micro_gain": micro_gain,
            "hard_gates": {
                "pixel_macro_dice_min": args.pixel_macro_floor,
                "pixel_micro_dice_min": args.pixel_micro_floor,
            },
            "checks": checks,
        },
        "prompt_conflict_policy": {
            "validation_filtering": False,
            "validation_conflict_episodes": int(summary["joint_prompt_conflict_episodes"]),
            "validation_conflict_episode_rate": float(summary["joint_prompt_conflict_episode_rate"]),
            "stress_set_persisted": True,
            "inference_action": "abstain",
            "user_message": "Conflicting positive/negative prompts; adjust or separate the prompts and retry.",
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / f"final_candidate_{args.timestamp}"
    if output_dir.exists():
        raise FileExistsError(f"immutable freeze directory already exists: {output_dir}")
    manifest = build_manifest(args)
    output_dir.mkdir(parents=True)
    output_path = output_dir / "final_candidate.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
