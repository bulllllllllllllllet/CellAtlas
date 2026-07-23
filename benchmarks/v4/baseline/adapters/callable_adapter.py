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
        self.verified_checkpoints: dict[str, dict[str, str]] = {}
        checkpoint = config.get("checkpoint")
        if checkpoint:
            expected = config.get("checkpoint_sha256")
            actual = sha256_file(checkpoint)
            if not expected or actual != expected:
                raise RuntimeError(
                    f"{self.method_name} checkpoint SHA-256 mismatch: expected={expected} actual={actual}"
                )
            self.verified_checkpoints["checkpoint"] = {
                "path": str(checkpoint), "sha256": str(actual),
            }
        checkpoints = config.get("checkpoints")
        if checkpoints is not None:
            if not isinstance(checkpoints, dict) or not checkpoints:
                raise TypeError(f"{self.method_name} checkpoints must be a non-empty mapping")
            for name, spec in checkpoints.items():
                if not isinstance(spec, dict):
                    raise TypeError(f"{self.method_name} checkpoint {name!r} must be a mapping")
                path = spec.get("path")
                expected = spec.get("sha256")
                if not path or not expected:
                    raise ValueError(f"{self.method_name} checkpoint {name!r} requires path and sha256")
                actual = sha256_file(path)
                if actual != expected:
                    raise RuntimeError(
                        f"{self.method_name} checkpoint {name!r} SHA-256 mismatch: "
                        f"expected={expected} actual={actual}"
                    )
                self.verified_checkpoints[str(name)] = {
                    "path": str(path), "sha256": str(actual),
                }
        self.backend = load_symbol(str(factory_spec))(config)
        self.load_report = getattr(self.backend, "load_report", None)
        if not isinstance(self.load_report, dict):
            raise TypeError("external backend must expose a strict checkpoint load_report dict")
        if self.load_report.get("strict") is not True:
            raise RuntimeError("external backend load_report must declare strict=true")
        if self.load_report.get("missing") or self.load_report.get("unexpected"):
            raise RuntimeError(f"external checkpoint load mismatch: {self.load_report}")
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
        raw_mask = np.asarray(raw["binary_mask"])
        if raw_mask.dtype != np.bool_:
            raise TypeError(f"external binary_mask must already be bool, got {raw_mask.dtype}")
        candidates = dict(raw.get("candidates", {}))
        frozen_rule = str(self.config.get("candidate_selection", ""))
        if not frozen_rule or candidates.get("selection_rule") != frozen_rule:
            raise RuntimeError(
                f"candidate selection mismatch: frozen={frozen_rule!r} returned={candidates.get('selection_rule')!r}"
            )
        return Prediction(
            probability=np.asarray(raw["probability"], dtype=np.float32),
            binary_mask=raw_mask,
            status=str(raw["status"]), latency_ms=float(raw["latency_ms"]),
            peak_memory_mb=float(raw["peak_memory_mb"]), candidates=candidates,
        )

    def provenance(self) -> dict[str, Any]:
        result = super().provenance() | {
            "method": self.method_name, "factory": self.config["factory"],
            "strict_load_report": self.load_report,
            "candidate_selection": self.config.get("candidate_selection"),
        }
        if self.verified_checkpoints:
            if not self.config.get("weights_source") or not self.config.get("license"):
                raise ValueError(f"{self.method_name} requires weights_source and license provenance")
            result |= {"verified_checkpoints": self.verified_checkpoints,
                       "weights_source": self.config["weights_source"], "license": self.config["license"]}
        return result
