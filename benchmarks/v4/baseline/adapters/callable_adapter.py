"""Strict bridge for third-party repositories with nonstandard Python APIs."""
from __future__ import annotations

from typing import Any
import inspect
import numpy as np

from .base import BaselineAdapter, EpisodeRequest, Prediction, load_symbol, sha256_file


class StrictCallableAdapter(BaselineAdapter):
    """Load an explicit factory and require the unified baseline signature.

    The factory receives the complete adapter config and returns an object with
    ``predict(image_or_multiscale_inputs, positive_prompts, negative_prompts,
    prompt_type, positive_box)``. No API guessing or compatibility fallback is
    performed.
    """

    method_name = "external"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        factory_spec = config.get("factory")
        if not factory_spec:
            raise ValueError(f"{self.method_name} requires an explicit factory module:symbol")
        checkpoint = config.get("checkpoint")
        if checkpoint:
            expected = config.get("checkpoint_sha256")
            actual = sha256_file(checkpoint)
            if not expected or actual != expected:
                raise RuntimeError(
                    f"{self.method_name} checkpoint SHA-256 mismatch: expected={expected} actual={actual}"
                )
        self.backend = load_symbol(str(factory_spec))(config)
        method = getattr(self.backend, "predict", None)
        if not callable(method):
            raise TypeError("external factory must return an object with predict()")
        expected_names = (
            "image_or_multiscale_inputs", "positive_prompts", "negative_prompts",
            "prompt_type", "positive_box",
        )
        actual_names = tuple(inspect.signature(method).parameters)
        if actual_names != expected_names:
            raise TypeError(f"external predict signature must be {expected_names}, got {actual_names}")

    def predict(self, request: EpisodeRequest) -> Prediction:
        model_input = request.multiscale_inputs if request.multiscale_inputs is not None else request.image
        raw = self.backend.predict(
            model_input,
            np.asarray(request.positive_points, dtype=np.float32),
            np.asarray(request.negative_points, dtype=np.float32),
            request.prompt_type,
            None if request.positive_box is None else np.asarray(request.positive_box, dtype=np.float32),
        )
        if isinstance(raw, Prediction):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("external predict must return Prediction or a dict")
        required = {"probability", "binary_mask", "status", "latency_ms", "peak_memory_mb"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"external prediction misses fields {sorted(missing)}")
        return Prediction(
            probability=np.asarray(raw["probability"], dtype=np.float32),
            binary_mask=np.asarray(raw["binary_mask"], dtype=bool),
            status=str(raw["status"]), latency_ms=float(raw["latency_ms"]),
            peak_memory_mb=float(raw["peak_memory_mb"]), candidates=dict(raw.get("candidates", {})),
        )

    def provenance(self) -> dict[str, Any]:
        result = super().provenance() | {"method": self.method_name, "factory": self.config["factory"]}
        if self.config.get("checkpoint"):
            result |= {"checkpoint": self.config["checkpoint"], "checkpoint_sha256": self.config["checkpoint_sha256"]}
        return result

