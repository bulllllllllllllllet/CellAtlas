from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _module(name: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", "-m", name, *arguments]


def build_stages(output_root: Path) -> list[tuple[str, list[str]]]:
    pilot = output_root / "manifests" / "pilot_5.csv"
    main = output_root / "manifests" / "main_20.csv"
    common = ["--output_root", str(output_root)]
    return [
        ("validate_setup_latest", _module("benchmarks.gdph_v2.validate_setup", *common)),
        (
            "validate_model_compatibility_latest",
            _module("benchmarks.gdph_v2.validate_model_compatibility", *common),
        ),
        ("audit_after_pilot_inference", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "polygon_labels_pilot",
            _module(
                "benchmarks.gdph_v2.polygon_labels",
                "--manifest", str(pilot), *common,
            ),
        ),
        ("audit_after_pilot_labels", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "paired_scale_validation",
            _module(
                "benchmarks.gdph_v2.eval_nucleus_detection",
                "--manifest", str(pilot), *common,
            ),
        ),
        ("audit_after_scale_validation", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "fullres_main20",
            _module(
                "benchmarks.gdph_v2.fullres_inference",
                "--manifest", str(main), *common,
            ),
        ),
        (
            "polygon_labels_main20",
            _module(
                "benchmarks.gdph_v2.polygon_labels",
                "--manifest", str(main), *common,
            ),
        ),
        (
            "linear_probe_5fold",
            _module(
                "benchmarks.gdph_v2.eval_crossval",
                "--manifest", str(main), *common,
            ),
        ),
        ("audit_after_linear_probe", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "generate_region_queries",
            _module(
                "benchmarks.gdph_v2.generate_queries",
                "--manifest", str(main), *common,
            ),
        ),
        ("evaluate_cell_retrieval", _module("benchmarks.gdph_v2.eval_retrieval", *common)),
        ("audit_after_region_retrieval", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "evaluate_cell_coverage",
            _module(
                "benchmarks.gdph_v2.eval_cell_coverage",
                "--manifest", str(main), *common,
            ),
        ),
        ("audit_after_cell_coverage", _module("benchmarks.gdph_v2.audit_experiment", *common)),
        (
            "patch_inference_main20",
            _module(
                "benchmarks.gdph_v2.patch_inference",
                "--manifest", str(main), *common,
            ),
        ),
        ("evaluate_patch_hybrid", _module("benchmarks.gdph_v2.eval_patch_retrieval", *common)),
        ("generate_final_report", _module("benchmarks.gdph_v2.generate_final_report", *common)),
        (
            "final_audit",
            _module(
                "benchmarks.gdph_v2.audit_experiment", *common, "--require_complete"
            ),
        ),
    ]


def run_stage(name: str, command: list[str], logs_dir: Path) -> None:
    log_path = logs_dir / f"pipeline_{name}.log"
    status_path = logs_dir / f"pipeline_{name}.status.json"
    started_at = datetime.now().astimezone().isoformat()
    status = {
        "stage": name,
        "state": "running",
        "started_at": started_at,
        "finished_at": None,
        "returncode": None,
        "command": command,
        "log_file": str(log_path),
    }
    _atomic_json(status_path, status)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{started_at}] START {' '.join(command)}\n")
        returncode = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
        finished_at = datetime.now().astimezone().isoformat()
        status.update(
            {
                "state": "completed" if returncode == 0 else "failed",
                "finished_at": finished_at,
                "returncode": returncode,
            }
        )
        _atomic_json(status_path, status)
        log.write(f"[{finished_at}] END returncode={returncode}\n")
    if returncode != 0:
        raise RuntimeError(f"pipeline stage failed: {name}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GDPH v2 stages 4-9 sequentially.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    stages = build_stages(output_root)
    if args.list:
        for name, command in stages:
            print(name, " ".join(command))
        return
    for name, command in stages:
        print(f"PIPELINE START {name}", flush=True)
        run_stage(name, command, output_root / "logs")
        print(f"PIPELINE COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
