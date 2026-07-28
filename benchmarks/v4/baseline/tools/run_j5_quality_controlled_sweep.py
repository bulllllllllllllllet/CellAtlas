#!/usr/bin/env python3
"""Run quality-controlled five-candidate J5 inference over WSI-class tasks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


CONFIG = Path("benchmarks/v4/phase_6_mask_decoder/configs/phase6_joint_j5_full_budget.yaml")
PHASE2_CONFIG = Path("benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml")
PHASE5_CONFIG = Path("benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml")
PHASE2_CHECKPOINT = Path(
    "/nfs-medical3/zyh/v4/phase2/runs/"
    "phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth"
)
CELL_CHECKPOINT = Path(
    "/nfs-medical3/zyh/v4/phase3/runs/"
    "phase_3_cell_region_train_20260719_025025/checkpoint_epoch_026.pth"
)
PHASE5_CHECKPOINT = Path(
    "/nfs-medical3/zyh/v4/phase5/runs/"
    "phase5_prompt_encoder_20260721_130650_formal/checkpoint_epoch_027.pth"
)
JOINT_CHECKPOINT = Path(
    "/nfs-medical3/zyh/v4/phase6/formal_full_runs/"
    "phase6_joint_pixel_20260722_142455/checkpoint_epoch_004.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--cell-feature-manifest",
        action="append",
        required=True,
        help="WSI_ID=/absolute/path/to/feature_index.parquet",
    )
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_feature_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --cell-feature-manifest value: {value}")
        wsi_id, raw_path = value.split("=", 1)
        if wsi_id in result:
            raise ValueError(f"duplicate feature manifest for {wsi_id}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[wsi_id] = path
    return result


def assign_jobs(frame: pd.DataFrame, gpus: list[int]) -> dict[int, list[dict[str, object]]]:
    assignments = {gpu: [] for gpu in gpus}
    loads = {gpu: 0 for gpu in gpus}
    for row in frame.sort_values(
        ["n_patch", "target_class", "wsi_id"], ascending=[False, True, True]
    ).to_dict("records"):
        gpu = min(gpus, key=lambda item: (loads[item], item))
        assignments[gpu].append(row)
        loads[gpu] += int(row["n_patch"])
    return assignments


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def run_gpu_worker(
    gpu: int,
    jobs: list[dict[str, object]],
    output: str,
    timestamp: str,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> dict[str, object]:
    root = Path(output)
    completed = root / f"completed_gpu{gpu}_{timestamp}.jsonl"
    failures = root / f"failures_gpu{gpu}_{timestamp}.jsonl"
    worker_log = root / "logs" / f"worker_gpu{gpu}_{timestamp}.log"
    successes = 0
    for task_index, job in enumerate(jobs):
        class_name = str(job["target_class_name"])
        wsi_id = str(job["wsi_id"])
        job_root = root / "runs" / class_name / wsi_id
        inference_dir = job_root / f"j5_multi_prompt_wsi_{timestamp}"
        job_log = root / "logs" / f"{class_name}__{wsi_id}__{timestamp}.log"
        if inference_dir.exists():
            raise FileExistsError(inference_dir)
        command = [
            sys.executable,
            "-m",
            "benchmarks.v4.whole_slide_inference.infer_wsi_multi_prompt",
            "--config",
            str(CONFIG),
            "--phase2-config",
            str(PHASE2_CONFIG),
            "--phase5-config",
            str(PHASE5_CONFIG),
            "--phase2-checkpoint",
            str(PHASE2_CHECKPOINT),
            "--cell-checkpoint",
            str(CELL_CHECKPOINT),
            "--phase5-checkpoint",
            str(PHASE5_CHECKPOINT),
            "--joint-checkpoint",
            str(JOINT_CHECKPOINT),
            "--tile-index",
            str(job["tile_index"]),
            "--cell-feature-manifest",
            str(job["cell_feature_manifest"]),
            "--candidate-manifest",
            str(job["candidate_manifest"]),
            "--wsi-id",
            wsi_id,
            "--output-root",
            str(job_root),
            "--timestamp",
            timestamp,
            "--gpu",
            str(gpu),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--threshold",
            str(threshold),
        ]
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8",
                "VIPS_CONCURRENCY": "4",
            }
        )
        started = datetime.now().isoformat()
        with job_log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd="/home/zhaoyh/CellAtlas",
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
            )
            launch = {
                "event": "job_start",
                "gpu": gpu,
                "compute_pid": process.pid,
                "task_index_on_gpu": task_index,
                "class": class_name,
                "wsi_id": wsi_id,
                "started": started,
                "command": command,
                "job_log": str(job_log),
            }
            append_jsonl(worker_log, launch)
            print(json.dumps(launch), flush=True)
            returncode = process.wait()
        finished = datetime.now().isoformat()
        metadata_path = inference_dir / "metadata.json"
        record = {
            "gpu": gpu,
            "compute_pid": process.pid,
            "class": class_name,
            "target_class": int(job["target_class"]),
            "wsi_id": wsi_id,
            "started": started,
            "finished": finished,
            "returncode": returncode,
            "inference_dir": str(inference_dir),
            "metadata": str(metadata_path),
            "job_log": str(job_log),
        }
        if returncode == 0 and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["decoded_candidate_count"] = int(metadata["decoded_candidate_count"])
            record["requested_candidate_count"] = int(metadata["requested_candidate_count"])
            if metadata["status"] != "complete" or metadata["requested_candidate_count"] != 5:
                record["error"] = "invalid completed metadata"
                append_jsonl(failures, record)
                raise RuntimeError(json.dumps(record))
            append_jsonl(completed, record)
            successes += 1
            print(json.dumps({"event": "job_complete", **record}), flush=True)
        else:
            record["error"] = "subprocess failed or metadata missing"
            append_jsonl(failures, record)
            print(json.dumps({"event": "job_failed", **record}), flush=True)
            raise RuntimeError(json.dumps(record))
    return {"gpu": gpu, "successes": successes, "assigned": len(jobs)}


def main() -> None:
    args = parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus contains duplicates")
    feature_map = parse_feature_map(args.cell_feature_manifest)
    output = args.output_root / f"j5_quality_sweep_{args.timestamp}"
    if output.exists():
        raise FileExistsError(output)
    (output / "logs").mkdir(parents=True)

    metadata_path = args.candidate_root / "metadata.json"
    candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not candidate_metadata.get("dice_blind"):
        raise ValueError("candidate metadata is not marked dice_blind")
    class_manifests = {
        str(name): Path(path)
        for name, path in candidate_metadata["class_manifests"].items()
    }

    tasks = pd.concat(
        [pd.read_parquet(path) for path in args.task_manifest],
        ignore_index=True,
    )
    if len(tasks) != 60 or tasks.duplicated(["wsi_id", "target_class_name"]).any():
        raise ValueError("expected 60 unique WSI-class tasks")
    missing_features = sorted(set(tasks["wsi_id"]) - set(feature_map))
    if missing_features:
        raise ValueError(f"missing cell feature manifests for {missing_features}")
    tasks["cell_feature_manifest"] = tasks["wsi_id"].map(
        lambda value: str(feature_map[str(value)])
    )
    tasks["candidate_manifest"] = tasks["target_class_name"].map(
        lambda value: str(class_manifests[str(value)])
    )
    tasks["run_output_root"] = tasks.apply(
        lambda row: str(output / "runs" / row["target_class_name"] / row["wsi_id"]),
        axis=1,
    )
    run_manifest = output / f"run_manifest_{args.timestamp}.parquet"
    tasks.to_parquet(run_manifest, index=False)
    assignments = assign_jobs(tasks, args.gpus)
    assignment_summary = {
        str(gpu): {
            "tasks": len(jobs),
            "tiles": sum(int(job["n_patch"]) for job in jobs),
        }
        for gpu, jobs in assignments.items()
    }
    run_metadata = {
        "timestamp": args.timestamp,
        "candidate_root": str(args.candidate_root),
        "candidate_metadata": str(metadata_path),
        "task_manifests": [str(path) for path in args.task_manifest],
        "run_manifest": str(run_manifest),
        "gpus": args.gpus,
        "assignment": assignment_summary,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "threshold": args.threshold,
        "checkpoints": {
            "phase2": str(PHASE2_CHECKPOINT),
            "cell": str(CELL_CHECKPOINT),
            "phase5": str(PHASE5_CHECKPOINT),
            "joint": str(JOINT_CHECKPOINT),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "sweep_start", "output": str(output), **run_metadata}), flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(
                run_gpu_worker,
                gpu,
                jobs,
                str(output),
                args.timestamp,
                args.batch_size,
                args.num_workers,
                args.threshold,
            ): gpu
            for gpu, jobs in assignments.items()
        }
        for future in as_completed(futures):
            results.append(future.result())
    final = {"event": "sweep_complete", "output": str(output), "workers": results}
    (output / "complete.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
