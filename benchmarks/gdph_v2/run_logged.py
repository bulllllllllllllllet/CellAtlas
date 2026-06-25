from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a durable benchmark stage with log/status files.")
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--status_file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("a command is required after --")
    log_path = Path(args.log_file)
    status_path = Path(args.status_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    status = {
        "state": "running",
        "started_at": started_at,
        "finished_at": None,
        "pid": None,
        "returncode": None,
        "command": command,
        "log_file": str(log_path),
    }
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{started_at}] START {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        status["pid"] = process.pid
        _write_status(status_path, status)
        returncode = process.wait()
        finished_at = datetime.now().astimezone().isoformat()
        status.update(
            {
                "state": "completed" if returncode == 0 else "failed",
                "finished_at": finished_at,
                "returncode": returncode,
            }
        )
        _write_status(status_path, status)
        log.write(f"[{finished_at}] END returncode={returncode}\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()

