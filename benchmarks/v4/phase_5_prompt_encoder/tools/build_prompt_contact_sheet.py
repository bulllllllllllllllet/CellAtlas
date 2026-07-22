#!/usr/bin/env python3
"""Rebuild a labeled contact sheet from an existing prompt audit artifact."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from benchmarks.v4.phase_5_prompt_encoder.tools.visualize_prompt_episodes import make_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    records = [
        json.loads(line)
        for line in (args.artifact_root / "completed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda item: int(item["target_class"]))
    output = args.artifact_root / f"prompt_contact_sheet_labeled_{stamp}.png"
    if output.exists():
        raise FileExistsError(output)
    make_contact_sheet(records, output)
    print(json.dumps({"event": "contact_sheet_complete", "episodes": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()

