#!/usr/bin/env python3
"""Run a smoke-gated, one-process-per-GPU full-WSI XCell feature batch."""
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


REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--smoke-rank", type=int)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--cell-batch-size", type=int, default=255)
    parser.add_argument("--max-cells", type=int, default=255)
    parser.add_argument("--spatial-grid-size", type=int, default=8)
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def case_root(batch_root: Path, row: dict, stamp: str) -> Path:
    return batch_root / "features" / f"{row['case_id']}_{stamp}"


def feature_root(batch_root: Path, row: dict, stamp: str) -> Path:
    return case_root(batch_root, row, stamp) / f"xcell_features_test_{stamp}"


def run_case(
    row: dict, gpu: int, args: argparse.Namespace, completed_stream
) -> dict:
    root = case_root(args.batch_root, row, args.timestamp)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite: {root}")
    root.mkdir(parents=True)
    status = {
        "selection_rank": int(row["selection_rank"]),
        "case_id": str(row["case_id"]),
        "wsi_id": str(row["wsi_id"]),
        "gpu": gpu,
        "status": "running",
        "tile_count": int(row["tile_count"]),
        "tile_index": str(row["tile_index"]),
        "case_root": str(root),
    }
    save_json(root / "status.json", status)
    command = [
        sys.executable,
        "-m",
        "benchmarks.v4.phase_3_cell_region.tools.extract_xcell_features",
        "--config", str(args.config),
        "--patch-index", str(row["tile_index"]),
        "--output-root", str(root),
        "--split", "test",
        "--start", "0",
        "--end", str(int(row["tile_count"])),
        "--shard-size", str(args.shard_size),
        "--max-cells", str(args.max_cells),
        "--cell-batch-size", str(args.cell_batch_size),
        "--preprocess-mode", "batched",
        "--selection-policy", "spatial_stratified",
        "--spatial-grid-size", str(args.spatial_grid_size),
        "--device", f"cuda:{gpu}",
        "--timestamp", args.timestamp,
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
    log_path = root / f"extract_{args.timestamp}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stream.write(f"EXIT_CODE={result.returncode}\n")
    if result.returncode != 0:
        status["status"] = "failed"
        status["exit_code"] = result.returncode
        save_json(root / "status.json", status)
        raise RuntimeError(f"feature extractor exited {result.returncode}: {row['wsi_id']}")
    status["status"] = "complete"
    status["feature_root"] = str(feature_root(args.batch_root, row, args.timestamp))
    save_json(root / "status.json", status)
    completed_stream.write(json.dumps(status, ensure_ascii=False) + "\n")
    completed_stream.flush()
    print(json.dumps({"event": "case_complete", **status}, ensure_ascii=False), flush=True)
    return status


def validate_smoke(row: dict, args: argparse.Namespace) -> dict:
    output = feature_root(args.batch_root, row, args.timestamp)
    metadata_path = output / "metadata.json"
    feature_index_path = output / "feature_index.parquet"
    failures_path = output / "failures.jsonl"
    checks = {
        "metadata_present": metadata_path.is_file(),
        "feature_index_present": feature_index_path.is_file(),
        "failures_present": failures_path.is_file(),
    }
    metadata = json.loads(metadata_path.read_text()) if checks["metadata_present"] else {}
    refs = pd.read_parquet(feature_index_path) if checks["feature_index_present"] else pd.DataFrame()
    checks.update(
        {
            "row_count_complete": int(metadata.get("rows", -1)) == int(row["tile_count"]),
            "indexed_rows_complete": (
                not refs.empty and int(refs["rows"].sum()) == int(row["tile_count"])
            ),
            "all_shards_present": (
                not refs.empty and all(Path(path).is_file() for path in refs["shard_path"])
            ),
            "failures_empty": (
                checks["failures_present"] and not failures_path.read_text().strip()
            ),
            "batched_preprocessing": metadata.get("preprocess_mode") == "batched",
            "spatial_selection": metadata.get("selection_policy") == "spatial_stratified",
            "split_test": metadata.get("split") == "test",
        }
    )
    source_indices: list[int] = []
    if checks["all_shards_present"]:
        for path in refs["shard_path"]:
            source_indices.extend(
                pd.read_parquet(path, columns=["source_index"])["source_index"].tolist()
            )
    checks["source_index_exact"] = source_indices == list(range(int(row["tile_count"])))
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "selection_rank": int(row["selection_rank"]),
        "wsi_id": str(row["wsi_id"]),
        "tile_count": int(row["tile_count"]),
        "checks": checks,
        "feature_root": str(output),
    }
    save_json(args.batch_root / f"smoke_validation_{args.timestamp}.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"smoke validation failed: {checks}")
    return result


def main() -> None:
    args = parse_args()
    if not args.batch_root.is_dir():
        raise FileNotFoundError(args.batch_root)
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("gpus must be a non-empty unique list")
    frame = pd.read_parquet(args.manifest).sort_values("selection_rank")
    if frame.empty or frame["selection_rank"].duplicated().any():
        raise ValueError("manifest is empty or has duplicate selection ranks")
    for path in frame["tile_index"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    smoke_rank = (
        int(args.smoke_rank)
        if args.smoke_rank is not None
        else int(frame.sort_values(["tile_count", "selection_rank"]).iloc[0]["selection_rank"])
    )
    smoke_matches = frame.loc[frame["selection_rank"].eq(smoke_rank)]
    if len(smoke_matches) != 1:
        raise ValueError(f"smoke rank {smoke_rank} does not identify one WSI")
    features_root = args.batch_root / "features"
    if features_root.exists():
        raise FileExistsError(f"refusing to overwrite: {features_root}")
    features_root.mkdir()
    shutil.copy2(args.manifest, args.batch_root / f"run_manifest_{args.timestamp}.parquet")
    run_record = {
        "timestamp": args.timestamp,
        "status": "running",
        "source_manifest": str(args.manifest),
        "batch_root": str(args.batch_root),
        "gpus": args.gpus,
        "wsi_count": len(frame),
        "tile_total": int(frame["tile_count"].sum()),
        "smoke_rank": smoke_rank,
        "preprocess_mode": "batched",
        "selection_policy": "spatial_stratified",
        "spatial_grid_size": args.spatial_grid_size,
        "max_cells": args.max_cells,
        "cell_batch_size": args.cell_batch_size,
        "protocol": "full overlapping 10x WSI grid; feature extraction only",
    }
    save_json(args.batch_root / f"run_record_{args.timestamp}.json", run_record)

    smoke_row = smoke_matches.iloc[0].to_dict()
    smoke_gpu = args.gpus[-1]
    with (args.batch_root / f"completed_gpu{smoke_gpu}_{args.timestamp}.jsonl").open(
        "a", encoding="utf-8"
    ) as completed:
        run_case(smoke_row, smoke_gpu, args, completed)
    smoke_result = validate_smoke(smoke_row, args)
    print(json.dumps({"event": "smoke_gate_passed", **smoke_result}), flush=True)

    work: queue.Queue[dict] = queue.Queue()
    remaining = frame.loc[~frame["selection_rank"].eq(smoke_rank)].sort_values(
        ["tile_count", "selection_rank"], ascending=[False, True]
    )
    for row in remaining.to_dict("records"):
        work.put(row)
    errors: list[dict] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        complete_path = args.batch_root / f"completed_gpu{gpu}_{args.timestamp}.jsonl"
        failure_path = args.batch_root / f"failures_gpu{gpu}_{args.timestamp}.jsonl"
        with complete_path.open("a", encoding="utf-8") as completed, failure_path.open(
            "a", encoding="utf-8"
        ) as failures:
            while True:
                try:
                    row = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_case(row, gpu, args, completed)
                except Exception as exc:
                    failure = {
                        "selection_rank": int(row["selection_rank"]),
                        "wsi_id": str(row["wsi_id"]),
                        "gpu": gpu,
                        "error": repr(exc),
                    }
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failures.flush()
                    with lock:
                        errors.append(failure)
                    print(json.dumps({"event": "case_failed", **failure}), flush=True)
                finally:
                    work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed_ranks: set[int] = set()
    for path in args.batch_root.glob("completed_gpu*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ranks.add(int(json.loads(line)["selection_rank"]))
    summary = {
        "status": (
            "complete"
            if not errors and len(completed_ranks) == len(frame)
            else "incomplete"
        ),
        "expected_wsi_count": len(frame),
        "completed_wsi_count": len(completed_ranks),
        "failed_wsi_count": len(errors),
        "failures": errors,
        "tile_total": int(frame["tile_count"].sum()),
    }
    save_json(args.batch_root / f"batch_summary_{args.timestamp}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
