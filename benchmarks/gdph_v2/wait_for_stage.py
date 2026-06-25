from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


TERMINAL_STATES = {"completed", "failed"}


def wait_for_completed(status_path: Path, poll_seconds: float) -> dict:
    """Wait for a run_logged status file and require successful completion."""
    while True:
        if status_path.exists():
            try:
                with open(status_path, "r", encoding="utf-8") as file:
                    status = json.load(file)
            except (OSError, json.JSONDecodeError):
                status = None
            if status and status.get("state") in TERMINAL_STATES:
                if status["state"] != "completed" or status.get("returncode") != 0:
                    raise RuntimeError(
                        f"dependency did not complete successfully: {status_path}: {status}"
                    )
                return status
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for a successful run_logged stage, then run a command."
    )
    parser.add_argument("--status_file", required=True)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("a command is required after --")
    if args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    wait_for_completed(Path(args.status_file), args.poll_seconds)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
