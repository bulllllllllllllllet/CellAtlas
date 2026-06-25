from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CUDA before starting a long GDPH stage.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA device after CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": available,
        "device_count": count,
        "requested_device": args.device,
        "device_name": torch.cuda.get_device_name(args.device)
        if available and args.device < count
        else None,
        "kernel_driver": _command_output(["cat", "/proc/driver/nvidia/version"]),
        "installed_module": _command_output(["modinfo", "nvidia"]),
    }
    report["passed"] = available and args.device < count
    output_path = Path(args.output_root) / "config" / "gpu_preflight.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("GPU preflight failed; do not start a long experiment")


if __name__ == "__main__":
    main()

