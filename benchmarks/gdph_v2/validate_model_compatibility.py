from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from new_inference_stream import inference as inference_base
from XCellFormer import XCellFormer


def _mapped_ctranspath_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state.items():
        new_key = key
        if "layers.0.downsample" in key:
            new_key = key.replace("layers.0.downsample", "layers.1.downsample")
        elif "layers.1.downsample" in key:
            new_key = key.replace("layers.1.downsample", "layers.2.downsample")
        elif "layers.2.downsample" in key:
            new_key = key.replace("layers.2.downsample", "layers.3.downsample")
        mapped[new_key] = value
    return mapped


def _compare(model_state: dict, checkpoint_state: dict) -> dict:
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    shared = model_keys & checkpoint_keys
    return {
        "model_keys": len(model_keys),
        "checkpoint_keys": len(checkpoint_keys),
        "missing": sorted(model_keys - checkpoint_keys),
        "unexpected": sorted(checkpoint_keys - model_keys),
        "shape_mismatch": sorted(
            key
            for key in shared
            if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen model/checkpoint compatibility.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--xcell_checkpoint",
        default="experiments/crc012_holdout3/20260515_160711/he_model_best.pth",
    )
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    args = parser.parse_args()
    device = torch.device("cpu")

    xcell = XCellFormer(
        input_dim=768,
        hidden_dim=512,
        n_heads=8,
        num_layers=4,
        output_dim=64,
        max_cells=255,
        use_large_vit=False,
        device="cpu",
    )
    xcell_checkpoint = torch.load(args.xcell_checkpoint, map_location="cpu")
    if isinstance(xcell_checkpoint, dict) and "state_dict" in xcell_checkpoint:
        xcell_checkpoint = xcell_checkpoint["state_dict"]
    xcell_report = _compare(xcell.state_dict(), xcell_checkpoint)
    xcell_report["passed"] = not any(
        xcell_report[key] for key in ("missing", "unexpected", "shape_mismatch")
    )

    ctranspath = inference_base.load_ctranspath(args.ctranspath_checkpoint, device)
    ctrans_checkpoint = torch.load(args.ctranspath_checkpoint, map_location="cpu")
    if isinstance(ctrans_checkpoint, dict) and "model" in ctrans_checkpoint:
        ctrans_checkpoint = ctrans_checkpoint["model"]
    ctrans_report = _compare(
        ctranspath.state_dict(), _mapped_ctranspath_state(ctrans_checkpoint)
    )
    allowed_unexpected = all(
        "relative_position_index" in key
        or "attn_mask" in key
        or key in {"patch_embed.norm.weight", "patch_embed.norm.bias"}
        for key in ctrans_report["unexpected"]
    )
    ctrans_report["unexpected_are_known_adapter_omissions"] = allowed_unexpected
    ctrans_report["passed"] = bool(
        not ctrans_report["missing"]
        and not ctrans_report["shape_mismatch"]
        and allowed_unexpected
    )

    report = {
        "device": "cpu",
        "xcellformer": xcell_report,
        "ctranspath": ctrans_report,
    }
    report["passed"] = xcell_report["passed"] and ctrans_report["passed"]
    output_path = Path(args.output_root) / "config" / "model_compatibility_validation.json"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise RuntimeError(f"model compatibility validation failed: {report}")


if __name__ == "__main__":
    main()
