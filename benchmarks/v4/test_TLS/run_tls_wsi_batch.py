#!/usr/bin/env python3
"""Run a gated, six-GPU TLS whole-slide J5 inference batch."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/test_TLS"))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--smoke-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run and validate the selected smoke WSI without launching the remaining batch.",
    )
    parser.add_argument(
        "--skip-smoke-gate", action="store_true",
        help="Launch the selected manifest rows directly; use only after an equivalent smoke is running or complete.",
    )
    parser.add_argument(
        "--exclude-ranks", type=int, nargs="*", default=[],
        help="Selection ranks already covered by a reusable case artifact.",
    )
    parser.add_argument(
        "--resume-batch", action="store_true",
        help="Resume and validate the existing timestamped batch root in place.",
    )
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--cellpose-net-batch-size", type=int, default=8)
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
        stream.write(f"EXIT_CODE={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def case_paths(batch_root: Path, row: dict, stamp: str) -> dict[str, Path]:
    case_root = batch_root / f"case_{int(row['selection_rank']):03d}_{row['case_id']}_{stamp}"
    return {
        "root": case_root,
        "prepared": case_root / f"tls_case_{stamp}",
        "features": case_root / f"tls_cell_features_{stamp}",
        "inference": case_root / f"wsi_inference_{stamp}",
    }


def completed_stage(stage: str, paths: dict[str, Path], row: dict) -> bool:
    if stage == "prepare":
        manifest = paths["prepared"] / "case_manifest.json"
        if not manifest.is_file():
            return False
        value = json.loads(manifest.read_text())
        return (
            int(value["tile_count"]) == int(row["tile_count"])
            and Path(value["wsi_path"]) == Path(row["wsi_path"])
            and Path(value["annotation_json"]) == Path(row["annotation_json"])
            and (paths["prepared"] / "wsi_tile_index.parquet").is_file()
            and (paths["prepared"] / "prompts.json").is_file()
            and (paths["prepared"] / "tls_gt_10x_pyramid.tif").is_file()
        )
    if stage == "features":
        metadata = paths["features"] / "metadata.json"
        index = paths["features"] / "feature_index.parquet"
        if not metadata.is_file() or not index.is_file():
            return False
        value = json.loads(metadata.read_text())
        return bool(value.get("complete_tile_index")) and int(value["rows"]) == int(row["tile_count"])
    if stage == "inference":
        metadata = paths["inference"] / "metadata.json"
        if not metadata.is_file():
            return False
        value = json.loads(metadata.read_text())
        return (
            value.get("status") == "complete"
            and bool(value.get("mask_returned"))
            and Path(value["probability_tiff"]).is_file()
            and Path(value["mask_tiff"]).is_file()
            and Path(value["coverage_tiff"]).is_file()
        )
    if stage == "evaluate":
        return len(list(paths["inference"].glob("tls_retrieval_report_*.json"))) == 1
    raise ValueError(f"unknown stage {stage}")


def execute_case(
    row: dict, gpu: int, batch_root: Path, stamp: str,
    args: argparse.Namespace, status_stream,
) -> dict:
    paths = case_paths(batch_root, row, stamp)
    if paths["root"].exists() and not args.resume_batch:
        raise FileExistsError(paths["root"])
    paths["root"].mkdir(parents=True, exist_ok=args.resume_batch)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "VIPS_CONCURRENCY": "4",
    })
    python = sys.executable
    positive_indices = [
        str(int(index)) for index in json.loads(row["positive_polygon_indices_json"])
    ]
    feature_command = [
        python, "benchmarks/v4/test_TLS/extract_tls_cell_features.py",
        "--tile-index", str(paths["prepared"] / "wsi_tile_index.parquet"),
        "--output-root", str(paths["root"]), "--device", f"cuda:{gpu}",
        "--xcell-batch-size", "8",
        "--tile-batch-size", str(args.tile_batch_size),
        "--cellpose-net-batch-size", str(args.cellpose_net_batch_size),
        "--timestamp", stamp,
    ]
    if args.resume_batch and paths["features"].exists():
        feature_command.append("--resume")
    stages = [
        ("prepare", [
            python, "benchmarks/v4/test_TLS/prepare_tls_case.py",
            "--wsi-path", str(row["wsi_path"]),
            "--annotation-json", str(row["annotation_json"]),
            "--wsi-id", str(row["wsi_id"]),
            "--positive-polygons", *positive_indices,
            "--negative-count", "3", "--level0-downsample", "4",
            "--tile-size", "512", "--stride", "384", "--split", "test",
            "--output-root", str(paths["root"]), "--timestamp", stamp,
        ]),
        ("features", feature_command),
        ("inference", [
            python, "-m", "benchmarks.v4.whole_slide_inference.infer_wsi",
            "--config", "benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml",
            "--phase2-config", "benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml",
            "--phase5-config", "benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml",
            "--phase2-checkpoint", "/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth",
            "--cell-checkpoint", "/nfs-medical3/zyh/v4/phase3/runs/phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth",
            "--phase5-checkpoint", "/nfs-medical3/zyh/v4/phase5/runs/phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth",
            "--joint-checkpoint", "/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth",
            "--tile-index", str(paths["prepared"] / "wsi_tile_index.parquet"),
            "--cell-feature-manifest", str(paths["features"] / "feature_index.parquet"),
            "--prompt-json", str(paths["prepared"] / "prompts.json"),
            "--gpus", str(gpu), "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers), "--seed", str(args.seed),
            "--output-root", str(paths["root"]), "--timestamp", stamp,
        ]),
        ("evaluate", [
            python, "benchmarks/v4/test_TLS/evaluate_tls_retrieval.py",
            "--inference-dir", str(paths["inference"]),
            "--case-manifest", str(paths["prepared"] / "case_manifest.json"),
            "--annotation-json", str(row["annotation_json"]), "--timestamp", stamp,
        ]),
    ]
    status_path = paths["root"] / "status.json"
    if args.resume_batch and status_path.is_file():
        status = json.loads(status_path.read_text())
        status.update({"gpu": gpu, "status": "running"})
        status["stages_complete"] = []
    else:
        status = {
            "selection_rank": int(row["selection_rank"]), "case_id": str(row["case_id"]),
            "wsi_id": str(row["wsi_id"]), "gpu": gpu, "status": "running",
            "case_root": str(paths["root"]), "stages_complete": [],
        }
    save_json(paths["root"] / "status.json", status)
    for stage, command in stages:
        if args.resume_batch and completed_stage(stage, paths, row):
            status["stages_complete"].append(stage)
            save_json(paths["root"] / "status.json", status)
            print(json.dumps({
                "event": "stage_reused", "rank": status["selection_rank"],
                "case_id": status["case_id"], "gpu": gpu, "stage": stage,
            }), flush=True)
            continue
        if stage == "inference" and paths["inference"].exists():
            raise RuntimeError(
                f"incomplete inference directory cannot be resumed safely: {paths['inference']}"
            )
        print(json.dumps({
            "event": "stage_started", "rank": status["selection_rank"],
            "case_id": status["case_id"], "gpu": gpu, "stage": stage,
        }), flush=True)
        run_command(command, paths["root"] / f"{stage}_{stamp}.log", env)
        status["stages_complete"].append(stage)
        save_json(paths["root"] / "status.json", status)
    status["status"] = "complete"
    save_json(paths["root"] / "status.json", status)
    status_stream.write(json.dumps(status, ensure_ascii=False) + "\n")
    status_stream.flush()
    print(json.dumps({"event": "case_complete", **status}), flush=True)
    return status


def validate_smoke(row: dict, batch_root: Path, stamp: str) -> dict:
    paths = case_paths(batch_root, row, stamp)
    case = json.loads((paths["prepared"] / "case_manifest.json").read_text())
    features = json.loads((paths["features"] / "metadata.json").read_text())
    inference = json.loads((paths["inference"] / "metadata.json").read_text())
    reports = sorted(paths["inference"].glob("tls_retrieval_report_*.json"))
    completed_rows = sum(1 for _ in (paths["inference"] / "completed.jsonl").open())
    feature_failures = (paths["features"] / "failures.jsonl").read_text().strip()
    inference_failures = (paths["inference"] / "failures.jsonl").read_text().strip()
    checks = {
        "case_tile_count_matches_manifest": int(case["tile_count"]) == int(row["tile_count"]),
        "feature_rows_complete": bool(features["complete_tile_index"])
        and int(features["rows"]) == int(row["tile_count"]),
        "inference_complete": inference.get("status") == "complete"
        and bool(inference.get("mask_returned")),
        "completed_tile_count": completed_rows == int(row["tile_count"]),
        "feature_failures_empty": not feature_failures,
        "inference_failures_empty": not inference_failures,
        "evaluation_report_present": len(reports) == 1,
        "split_is_test": case.get("split") == "test" and inference.get("split") == "test",
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "selection_rank": int(row["selection_rank"]), "case_id": str(row["case_id"]),
        "tile_count": int(row["tile_count"]), "checks": checks,
    }
    save_json(batch_root / f"smoke_validation_{stamp}.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"smoke validation failed: {checks}")
    return result


def main() -> None:
    args = parse_args()
    if len(args.gpus) != len(set(args.gpus)) or not args.gpus:
        raise ValueError("gpus must be a non-empty unique list")
    source_frame = pd.read_parquet(args.manifest).sort_values("selection_rank")
    if source_frame.empty or source_frame.selection_rank.duplicated().any():
        raise ValueError("manifest must contain unique selection ranks")
    if len(args.exclude_ranks) != len(set(args.exclude_ranks)):
        raise ValueError("exclude-ranks must be unique")
    missing_exclusions = sorted(set(args.exclude_ranks) - set(source_frame.selection_rank))
    if missing_exclusions:
        raise ValueError(f"excluded ranks absent from manifest: {missing_exclusions}")
    frame = source_frame.loc[~source_frame.selection_rank.isin(args.exclude_ranks)].copy()
    if frame.empty:
        raise ValueError("no manifest rows remain after exclusions")
    smoke_matches = frame.loc[frame.selection_rank.eq(args.smoke_rank)]
    if not args.skip_smoke_gate and len(smoke_matches) != 1:
        raise ValueError("smoke-rank must identify exactly one included manifest row")
    if args.smoke_only and args.skip_smoke_gate:
        raise ValueError("smoke-only and skip-smoke-gate are mutually exclusive")
    batch_root = args.output_root / f"tls30_j5_inference_{args.timestamp}"
    if args.resume_batch:
        if not batch_root.is_dir():
            raise FileNotFoundError(batch_root)
    else:
        if batch_root.exists():
            raise FileExistsError(batch_root)
        batch_root.mkdir(parents=True)
        shutil.copy2(args.manifest, batch_root / f"tls_wsi_manifest_{args.timestamp}.parquet")
    run_record = {
        "timestamp": args.timestamp, "source_manifest": str(args.manifest),
        "batch_root": str(batch_root), "gpus": args.gpus,
        "smoke_rank": args.smoke_rank, "batch_size": args.batch_size,
        "num_workers": args.num_workers, "seed": args.seed,
        "smoke_only": args.smoke_only,
        "skip_smoke_gate": args.skip_smoke_gate,
        "excluded_reusable_ranks": args.exclude_ranks,
        "resume_batch": args.resume_batch,
        "tile_batch_size": args.tile_batch_size,
        "cellpose_net_batch_size": args.cellpose_net_batch_size,
        "protocol": (
            "pre-registered deterministic half-instance positive TLS points "
            "+ 3 deterministic tissue negatives"
        ),
        "evaluation": "union, prompted polygon, and held-out TLS retrieval",
        "source_wsi_count": len(source_frame),
        "wsi_count": len(frame), "tile_total": int(frame.tile_count.sum()),
    }
    if args.resume_batch:
        launch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(batch_root / f"resume_manifest_{launch_stamp}.json", run_record)
    else:
        save_json(batch_root / f"run_manifest_{args.timestamp}.json", run_record)

    if not args.skip_smoke_gate:
        smoke_row = smoke_matches.iloc[0].to_dict()
        smoke_gpu = args.gpus[-1]
        smoke_status_path = batch_root / f"completed_gpu{smoke_gpu}_{args.timestamp}.jsonl"
        with smoke_status_path.open("a", encoding="utf-8") as stream:
            execute_case(smoke_row, smoke_gpu, batch_root, args.timestamp, args, stream)
        smoke_result = validate_smoke(smoke_row, batch_root, args.timestamp)
        print(json.dumps({"event": "smoke_gate_passed", **smoke_result}), flush=True)
        if args.smoke_only:
            save_json(batch_root / f"batch_summary_{args.timestamp}.json", {
                "status": "smoke_complete",
                "expected_wsi_count": len(frame),
                "completed_wsi_count": 1,
                "smoke_rank": args.smoke_rank,
                "formal_batch_launched": False,
            })
            return

    work: queue.Queue[dict] = queue.Queue()
    if args.skip_smoke_gate:
        remaining = frame
    else:
        remaining = frame.loc[~frame.selection_rank.eq(args.smoke_rank)]
    remaining = remaining.sort_values(
        ["tile_count", "selection_rank"], ascending=[False, True],
    )
    for row in remaining.to_dict("records"):
        work.put(row)
    errors: list[dict] = []
    error_lock = threading.Lock()

    def gpu_worker(gpu: int) -> None:
        completed_path = batch_root / f"completed_gpu{gpu}_{args.timestamp}.jsonl"
        failures_path = batch_root / f"failures_gpu{gpu}_{args.timestamp}.jsonl"
        with completed_path.open("a", encoding="utf-8") as completed, failures_path.open(
            "a", encoding="utf-8"
        ) as failures:
            while True:
                try:
                    row = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    execute_case(row, gpu, batch_root, args.timestamp, args, completed)
                except Exception as exc:
                    failure = {
                        "selection_rank": int(row["selection_rank"]),
                        "case_id": str(row["case_id"]), "gpu": gpu,
                        "error": repr(exc),
                    }
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failures.flush()
                    with error_lock:
                        errors.append(failure)
                    print(json.dumps({"event": "case_failed", **failure}), flush=True)
                finally:
                    work.task_done()

    threads = [threading.Thread(target=gpu_worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    completed_ranks = set()
    for path in batch_root.glob("completed_gpu*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ranks.add(int(json.loads(line)["selection_rank"]))
    expected_ranks = set(map(int, frame.selection_rank))
    completed_count = len(completed_ranks & expected_ranks)
    summary = {
        "status": "complete" if not errors and completed_count == len(frame) else "incomplete",
        "expected_wsi_count": len(frame), "completed_wsi_count": completed_count,
        "failed_wsi_count": len(errors), "failures": errors,
    }
    save_json(batch_root / f"batch_summary_{args.timestamp}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
