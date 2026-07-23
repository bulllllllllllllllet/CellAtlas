"""Strict integrations for the pinned SAM-Med2D and WSI-SAM repositories."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
import importlib
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch


def _verify_source(config: dict[str, Any]) -> tuple[Path, str]:
    root = Path(config["source_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"third-party source root does not exist: {root}")
    expected = str(config["source_commit"])
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"source commit mismatch for {root}: expected={expected} actual={actual}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"tracked third-party source files differ from HEAD: {root}")
    return root, actual


def _activate_source(root: Path, top_level: tuple[str, ...]) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    conflicts = [name for name in sys.modules if name.split(".", 1)[0] in top_level]
    if conflicts:
        locations = {name: getattr(sys.modules[name], "__file__", None) for name in conflicts}
        if any(path is None or not str(path).startswith(root_text) for path in locations.values()):
            raise RuntimeError(f"conflicting third-party modules already imported: {locations}")


def _state_dict(path: str | Path, key: str | None = None) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None:
        if not isinstance(payload, dict) or key not in payload:
            raise RuntimeError(f"checkpoint {path} does not contain key {key!r}")
        payload = payload[key]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"checkpoint {path} does not contain a non-empty state dict")
    return payload


def _strict_load(module: torch.nn.Module, state: dict[str, torch.Tensor], name: str) -> dict[str, Any]:
    incompat = module.load_state_dict(state, strict=True)
    return {
        "name": name,
        "strict": True,
        "loaded_tensors": len(state),
        "missing": list(incompat.missing_keys),
        "unexpected": list(incompat.unexpected_keys),
    }


class SAMMed2DBackend:
    def __init__(self, config: dict[str, Any]):
        root, commit = _verify_source(config)
        _activate_source(root, ("segment_anything",))
        build = importlib.import_module("segment_anything.build_sam")
        args = SimpleNamespace(
            image_size=int(config["image_size"]),
            sam_checkpoint=None,
            encoder_adapter=bool(config["encoder_adapter"]),
        )
        model_type = str(config["model_type"])
        if model_type not in build.sam_model_registry:
            raise ValueError(f"unsupported SAM-Med2D model_type {model_type!r}")
        model = build.sam_model_registry[model_type](args)
        report = _strict_load(model, _state_dict(config["checkpoint"], "model"), "sam_med2d")
        self.device = torch.device(str(config["device"]))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM-Med2D configuration requires CUDA, but CUDA is unavailable")
        self.model = model.to(self.device).eval()
        self.threshold = float(config["threshold"])
        self.selection_rule = str(config["candidate_selection"])
        self.load_report = {
            "strict": True, "missing": [], "unexpected": [],
            "components": [report], "source_root": str(root), "source_commit": commit,
        }

    def predict(
        self, image_or_multiscale_inputs, positive_prompts, negative_prompts,
        prompt_type, positive_box,
    ) -> dict[str, Any]:
        if isinstance(image_or_multiscale_inputs, dict):
            image = image_or_multiscale_inputs["fine"]
        else:
            image = image_or_multiscale_inputs
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        image = np.asarray(image, dtype=np.uint8)
        height, width = image.shape[:2]
        mean = self.model.pixel_mean.squeeze().detach().cpu().numpy()
        std = self.model.pixel_std.squeeze().detach().cpu().numpy()
        normalized = (image.astype(np.float32) - mean) / std
        image_tensor = torch.as_tensor(normalized).permute(2, 0, 1)[None].to(self.device)
        image_tensor = F.interpolate(
            image_tensor, size=(self.model.image_encoder.img_size,) * 2, mode="nearest",
        )
        points = np.concatenate([positive_prompts, negative_prompts], axis=0).astype(np.float32)
        labels = np.concatenate([
            np.ones(len(positive_prompts), dtype=np.int32),
            np.zeros(len(negative_prompts), dtype=np.int32),
        ])
        scale = np.asarray([
            self.model.image_encoder.img_size / width,
            self.model.image_encoder.img_size / height,
        ], dtype=np.float32)
        points_tensor = torch.as_tensor(points * scale, dtype=torch.float32, device=self.device)[None]
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64, device=self.device)[None]
        box_tensor = None
        if prompt_type != "point":
            box = np.asarray(positive_box, dtype=np.float32) * np.tile(scale, 2)
            box_tensor = torch.as_tensor(box, dtype=torch.float32, device=self.device)[None]
        with torch.inference_mode():
            features = self.model.image_encoder(image_tensor)
            sparse, dense = self.model.prompt_encoder(
                points=(points_tensor, labels_tensor), boxes=box_tensor, masks=None,
            )
            low_res_logits, quality = self.model.mask_decoder(
                image_embeddings=features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            logits_tensor = F.interpolate(
                low_res_logits,
                size=(self.model.image_encoder.img_size,) * 2,
                mode="bilinear", align_corners=False,
            )
            logits_tensor = F.interpolate(
                logits_tensor, size=(height, width), mode="bilinear", align_corners=False,
            )
        logits = logits_tensor[0].float().cpu().numpy()
        predicted_iou = quality[0].float().cpu().numpy()
        if logits.ndim != 3 or len(logits) != len(predicted_iou) or not np.isfinite(predicted_iou).all():
            raise RuntimeError("SAM-Med2D returned an invalid candidate set")
        selected = int(np.argmax(predicted_iou))
        probability = (1.0 / (1.0 + np.exp(-logits[selected]))).astype(np.float32)
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        return {
            "probability": probability,
            "binary_mask": probability >= self.threshold,
            "status": "completed",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "peak_memory_mb": float(peak),
            "candidates": {
                "selection_rule": self.selection_rule,
                "selected_index": selected,
                "predicted_iou": predicted_iou.tolist(),
                "candidate_count": int(len(logits)),
            },
        }


@contextmanager
def _bind_decoder_checkpoint(path: Path) -> Iterator[None]:
    original = torch.load

    def bound_load(requested, *args, **kwargs):
        if str(requested) == "pretrained_checkpoint/vit_tiny_maskdecoder.pt":
            requested = path
        return original(requested, *args, **kwargs)

    torch.load = bound_load
    try:
        yield
    finally:
        torch.load = original


class WSISAMBackend:
    def __init__(self, config: dict[str, Any]):
        root, commit = _verify_source(config)
        _activate_source(root, ("segment_anything_training", "network"))
        registry = importlib.import_module("segment_anything_training")
        network = importlib.import_module("network")
        checkpoints = config["checkpoints"]
        self.device = torch.device(str(config["device"]))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("WSI-SAM configuration requires CUDA, but CUDA is unavailable")
        self.sam = registry.sam_model_registry["vit_tiny"](
            checkpoint=str(checkpoints["mobile_sam"]["path"])
        )
        decoder_base = Path(checkpoints["vit_tiny_maskdecoder"]["path"])
        base_decoder = network.MaskDecoder(
            transformer_dim=256,
            transformer=network.TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
            ),
            num_multimask_outputs=3,
            activation=network.nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        reports = [
            _strict_load(base_decoder, _state_dict(decoder_base), "vit_tiny_maskdecoder"),
        ]
        with _bind_decoder_checkpoint(decoder_base):
            self.net_high = network.MaskDecoderHigh("vit_tiny")
            self.net_low = network.MaskDecoderLow("vit_tiny")
        reports += [
            _strict_load(self.net_high, _state_dict(checkpoints["net_high"]["path"], "model"), "net_high"),
            _strict_load(self.net_low, _state_dict(checkpoints["net_low"]["path"], "model"), "net_low"),
        ]
        self.sam = self.sam.to(self.device).eval()
        self.net_high = self.net_high.to(self.device).eval()
        self.net_low = self.net_low.to(self.device).eval()
        self.threshold = float(config["threshold"])
        self.selection_rule = str(config["candidate_selection"])
        self.load_report = {
            "strict": True, "missing": [], "unexpected": [], "components": reports,
            "source_root": str(root), "source_commit": commit,
            "mobile_sam_strict": True,
        }

    @staticmethod
    def _resize_image(image: np.ndarray, size: int = 1024) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1).float()[None]
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return tensor[0].round().clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _low_context(inputs: dict[str, Any]) -> np.ndarray:
        x, y, width, height = map(int, inputs["fine_box_level0"])
        if width != height:
            raise ValueError(f"WSI-SAM requires a square fine field, got {(width, height)}")
        center_x, center_y = map(float, inputs["center_level0"])
        low_width, low_height = 2 * width, 2 * height
        low_x, low_y = int(round(center_x - low_width / 2)), int(round(center_y - low_height / 2))
        return read_he_patch(Path(inputs["wsi_path"]), low_x, low_y, low_width, low_height, 1024)

    def _encode(self, image: torch.Tensor, points: np.ndarray, labels: np.ndarray, box: np.ndarray | None):
        record: dict[str, torch.Tensor] = {"image": image.to(self.device)}
        record["point_coords"] = torch.as_tensor(points, dtype=torch.float32, device=self.device)[None]
        record["point_labels"] = torch.as_tensor(labels, dtype=torch.int64, device=self.device)[None]
        if box is not None:
            record["boxes"] = torch.as_tensor(box, dtype=torch.float32, device=self.device)[None]
        return self.sam([record], multimask_output=False)

    def predict(
        self, image_or_multiscale_inputs, positive_prompts, negative_prompts,
        prompt_type, positive_box,
    ) -> dict[str, Any]:
        if not isinstance(image_or_multiscale_inputs, dict):
            raise TypeError("WSI-SAM requires the explicit multiscale input mapping")
        inputs = image_or_multiscale_inputs
        fine = np.asarray(inputs["fine"], dtype=np.uint8)
        original_shape = fine.shape[:2]
        high = self._resize_image(fine)
        low = self._resize_image(self._low_context(inputs))
        points_10x = np.concatenate([positive_prompts, negative_prompts], axis=0).astype(np.float32)
        labels = np.concatenate([
            np.ones(len(positive_prompts), dtype=np.int64),
            np.zeros(len(negative_prompts), dtype=np.int64),
        ])
        scale = np.asarray([1024 / original_shape[1], 1024 / original_shape[0]], dtype=np.float32)
        high_points = points_10x * scale
        low_points = high_points / 2.0 + 256.0
        high_box = None if prompt_type == "point" else np.asarray(positive_box, dtype=np.float32) * np.tile(scale, 2)
        low_box = None if high_box is None else high_box / 2.0 + 256.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            out_high, interm_high = self._encode(high, high_points, labels, high_box)
            out_low, interm_low = self._encode(low, low_points, labels, low_box)
            high_item, low_item = out_high[0], out_low[0]
            masks_sam_high, _, token_high, embedding_high = self.net_high(
                image_embeddings=high_item["encoder_embedding"], image_pe=[high_item["image_pe"]],
                sparse_prompt_embeddings=[high_item["sparse_embeddings"]],
                dense_prompt_embeddings=[high_item["dense_embeddings"]],
                multimask_output=False, wsi_token_only=False, interm_embeddings=interm_high,
            )
            _, _, token_low, _ = self.net_low(
                image_embeddings=low_item["encoder_embedding"], image_pe=[low_item["image_pe"]],
                sparse_prompt_embeddings=[low_item["sparse_embeddings"]],
                dense_prompt_embeddings=[low_item["dense_embeddings"]],
                multimask_output=False, wsi_token_only=False, interm_embeddings=interm_low,
            )
            fused = ((token_high + token_low) @ embedding_high).view(1, -1, 256, 256)
            logits = fused + masks_sam_high
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)[0, 0]
            probability = torch.sigmoid(logits).float().cpu().numpy()
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        return {
            "probability": probability,
            "binary_mask": probability >= self.threshold,
            "status": "completed",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "peak_memory_mb": float(peak),
            "candidates": {"selection_rule": self.selection_rule, "candidate_count": 1, "selected_index": 0},
        }


def create_sam_med2d(config: dict[str, Any]) -> SAMMed2DBackend:
    return SAMMed2DBackend(config)


def create_wsi_sam(config: dict[str, Any]) -> WSISAMBackend:
    return WSISAMBackend(config)
