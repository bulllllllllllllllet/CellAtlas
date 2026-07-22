#!/usr/bin/env python3
"""Wait for a stable set of idle GPUs, then run one command on that set."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-gpus", type=int, required=True)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--min-free-memory-mib", type=int, default=30000)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def gpu_state() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        index, used, total, utilization = [int(value.strip()) for value in line.split(",")]
        rows.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": total - used,
                "utilization_percent": utilization,
            }
        )
    return rows


def emit(event: str, **values) -> None:
    print(
        json.dumps(
            {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **values},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    previous: tuple[int, ...] | None = None
    consecutive = 0
    emit(
        "wait_start",
        required_gpus=args.required_gpus,
        max_utilization=args.max_utilization,
        min_free_memory_mib=args.min_free_memory_mib,
        stable_checks=args.stable_checks,
        poll_seconds=args.poll_seconds,
    )
    while True:
        rows = gpu_state()
        eligible = [
            row for row in rows
            if row["utilization_percent"] <= args.max_utilization
            and row["memory_free_mib"] >= args.min_free_memory_mib
        ]
        eligible.sort(key=lambda row: (row["utilization_percent"], -row["memory_free_mib"], row["index"]))
        candidate = tuple(sorted(row["index"] for row in eligible[: args.required_gpus]))
        if len(candidate) == args.required_gpus and candidate == previous:
            consecutive += 1
        elif len(candidate) == args.required_gpus:
            previous = candidate
            consecutive = 1
        else:
            previous = None
            consecutive = 0
        emit("gpu_poll", gpus=rows, candidate=list(candidate), consecutive_stable_checks=consecutive)
        if consecutive >= args.stable_checks:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, candidate))
            emit("launch", selected_gpus=list(candidate), command=args.command)
            completed = subprocess.run(args.command, env=environment)
            emit("command_exit", selected_gpus=list(candidate), exit_code=completed.returncode)
            return completed.returncode
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
