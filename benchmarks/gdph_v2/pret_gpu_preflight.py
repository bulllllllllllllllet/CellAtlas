from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def run_command(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "error": f"{command[0]} not found"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - diagnostics script
        return {"available": True, "command": command, "error": repr(exc)}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def torch_info() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"import_ok": False, "error": repr(exc)}
    output = {
        "import_ok": True,
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
                }
            )
        except Exception as exc:
            devices.append({"index": index, "error": repr(exc)})
    output["devices"] = devices
    try:
        if torch.cuda.is_available():
            x = torch.ones((1,), device="cuda")
            output["cuda_tensor_test"] = float(x.item())
        else:
            output["cuda_tensor_test"] = None
    except Exception as exc:
        output["cuda_tensor_test_error"] = repr(exc)
    return output


def main() -> None:
    report = {
        "python": sys.executable,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key, "")
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "CONDA_PREFIX",
                "PYTHONNOUSERSITE",
            ]
        },
        "dev_nodes": sorted(name for name in os.listdir("/dev") if name.startswith("nvidia")) if os.path.isdir("/dev") else [],
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvidia_smi_query": run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "torch": torch_info(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
