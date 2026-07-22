"""Official Segment Anything adapter with predicted-IoU candidate selection."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import BaselineAdapter, EpisodeRequest, Prediction, sha256_file


class SAMAdapter(BaselineAdapter):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("SAM dependency 'segment_anything' is unavailable") from exc
        checkpoint = config.get("checkpoint")
        expected = config.get("checkpoint_sha256")
        if not checkpoint or not expected:
            raise ValueError("SAM checkpoint and checkpoint_sha256 are required")
        actual = sha256_file(checkpoint)
        if actual != expected:
            raise RuntimeError(f"SAM checkpoint SHA-256 mismatch: expected={expected} actual={actual}")
        model_type = str(config.get("model_type", "vit_h"))
        if model_type not in sam_model_registry:
            raise ValueError(f"unknown SAM model_type {model_type!r}")
        device = str(config.get("device", "cuda:0"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM configuration requires CUDA, but CUDA is unavailable")
        self.device = torch.device(device)
        self.model = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(self.device).eval()
        self.predictor = SamPredictor(self.model)
        self.threshold = float(config.get("threshold", 0.5))

    def predict(self, request: EpisodeRequest) -> Prediction:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.predictor.set_image(request.image)
        points = np.concatenate([request.positive_points, request.negative_points], axis=0).astype(np.float32)
        labels = np.concatenate([
            np.ones(len(request.positive_points), dtype=np.int32),
            np.zeros(len(request.negative_points), dtype=np.int32),
        ])
        box = None if request.prompt_type == "point" else request.positive_box[None].astype(np.float32)
        masks, predicted_iou, low_res_logits = self.predictor.predict(
            point_coords=points, point_labels=labels, box=box, multimask_output=True,
            return_logits=False,
        )
        predicted_iou = np.asarray(predicted_iou, dtype=np.float32)
        if masks.ndim != 3 or len(masks) != len(predicted_iou) or not np.isfinite(predicted_iou).all():
            raise RuntimeError("SAM returned an invalid candidate set")
        selected = int(np.argmax(predicted_iou))
        # Official predictor returns thresholded masks. Preserve this deterministic
        # result as both binary output and a {0,1} probability map; never use GT.
        binary = np.asarray(masks[selected], dtype=bool)
        probability = binary.astype(np.float32)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0.0
        return Prediction(
            probability=probability, binary_mask=binary, peak_memory_mb=float(peak),
            candidates={
                "selection_rule": "argmax_predicted_iou", "selected_index": selected,
                "predicted_iou": predicted_iou.tolist(), "candidate_count": int(len(masks)),
                "low_res_logits_shape": list(np.asarray(low_res_logits).shape),
            },
        )

    def provenance(self) -> dict[str, Any]:
        return super().provenance() | {
            "method": "sam", "checkpoint": self.config["checkpoint"],
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "candidate_selection": "argmax_predicted_iou",
        }
