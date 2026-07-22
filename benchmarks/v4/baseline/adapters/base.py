"""Strict common contract shared by every baseline adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any
import hashlib
import time

import numpy as np


SUPPORTED_PROMPTS = frozenset({"point", "small", "large"})


@dataclass(frozen=True)
class EpisodeRequest:
    occurrence_id: str
    image: np.ndarray
    positive_points: np.ndarray
    negative_points: np.ndarray
    prompt_type: str
    positive_box: np.ndarray | None = None
    multiscale_inputs: dict[str, Any] | None = None

    def validate(self) -> None:
        image = np.asarray(self.image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"image must be uint8 RGB [H,W,3], got {image.shape} {image.dtype}")
        if self.prompt_type not in SUPPORTED_PROMPTS:
            raise ValueError(f"unsupported prompt type {self.prompt_type!r}")
        height, width = image.shape[:2]
        for name, points in (("positive", self.positive_points), ("negative", self.negative_points)):
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f"{name} points must be finite [N,2]")
            if ((points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)).any():
                raise ValueError(f"{name} prompt outside image bounds")
        if len(self.positive_points) == 0:
            raise ValueError("at least one positive prompt is required")
        if self.prompt_type != "point":
            if self.positive_box is None:
                raise ValueError(f"{self.prompt_type} requires frozen positive_box geometry")
            box = np.asarray(self.positive_box, dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("positive_box must be finite [x0,y0,x1,y1]")
            x0, y0, x1, y1 = box.tolist()
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"invalid positive_box {box.tolist()} for {(width, height)}")


@dataclass
class Prediction:
    probability: np.ndarray
    binary_mask: np.ndarray
    status: str = "completed"
    latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    candidates: dict[str, Any] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        probability = np.asarray(self.probability)
        mask = np.asarray(self.binary_mask)
        if probability.shape != shape or mask.shape != shape:
            raise ValueError(f"prediction shape mismatch: {probability.shape}, {mask.shape}, expected {shape}")
        if not np.isfinite(probability).all() or float(probability.min()) < 0 or float(probability.max()) > 1:
            raise ValueError("probability must be finite and in [0,1]")
        if mask.dtype != np.bool_:
            raise ValueError(f"binary_mask must have bool dtype, got {mask.dtype}")
        if self.status not in {"completed", "abstained"}:
            raise ValueError(f"invalid prediction status {self.status!r}")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not np.isfinite(self.peak_memory_mb) or self.peak_memory_mb < 0:
            raise ValueError("peak_memory_mb must be finite and non-negative")


class BaselineAdapter(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def predict(self, request: EpisodeRequest) -> Prediction:
        raise NotImplementedError

    def timed_predict(self, request: EpisodeRequest) -> Prediction:
        request.validate()
        started = time.perf_counter()
        prediction = self.predict(request)
        if prediction.latency_ms == 0:
            prediction.latency_ms = (time.perf_counter() - started) * 1000.0
        prediction.validate(request.image.shape[:2])
        return prediction

    def provenance(self) -> dict[str, Any]:
        return {"adapter": f"{type(self).__module__}:{type(self).__name__}"}


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_symbol(specification: str):
    if ":" not in specification:
        raise ValueError("factory must use 'module:symbol' syntax")
    module, symbol = specification.split(":", 1)
    return getattr(import_module(module), symbol)


def build_adapter(config: dict[str, Any]) -> BaselineAdapter:
    adapter = str(config.get("adapter", ""))
    mapping = {
        "fixed_slic": "benchmarks.v4.baseline.adapters.fixed_slic_adapter:FixedSLICAdapter",
        "sam": "benchmarks.v4.baseline.adapters.sam_adapter:SAMAdapter",
        "sam_med2d": "benchmarks.v4.baseline.adapters.sam_med2d_adapter:SAMMed2DAdapter",
        "wsi_sam": "benchmarks.v4.baseline.adapters.wsi_sam_adapter:WSISAMAdapter",
        "careprompt": "benchmarks.v4.baseline.adapters.careprompt_adapter:CaRePromptAdapter",
    }
    if adapter not in mapping:
        raise ValueError(f"unknown adapter {adapter!r}; expected one of {sorted(mapping)}")
    instance = load_symbol(mapping[adapter])(config)
    if not isinstance(instance, BaselineAdapter):
        raise TypeError(f"adapter {adapter} does not implement BaselineAdapter")
    return instance

