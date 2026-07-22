"""Deterministic Fixed-SLIC + colour prompt matching baseline."""
from __future__ import annotations

from typing import Any
import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic

from .base import BaselineAdapter, EpisodeRequest, Prediction


class FixedSLICAdapter(BaselineAdapter):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.n_segments = int(config.get("n_segments", 64))
        self.compactness = float(config.get("compactness", 10.0))
        self.temperature = float(config.get("temperature", 8.0))
        self.threshold = float(config.get("threshold", 0.5))
        if self.n_segments < 2 or self.compactness <= 0 or self.temperature <= 0:
            raise ValueError("invalid Fixed-SLIC parameters")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly in (0,1)")

    @staticmethod
    def _prompt_segments(labels: np.ndarray, points: np.ndarray) -> np.ndarray:
        rounded = np.rint(points).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, labels.shape[1] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, labels.shape[0] - 1)
        return np.unique(labels[rounded[:, 1], rounded[:, 0]])

    def predict(self, request: EpisodeRequest) -> Prediction:
        labels = slic(
            request.image, n_segments=self.n_segments, compactness=self.compactness,
            start_label=0, channel_axis=-1, convert2lab=True,
        ).astype(np.int32)
        lab = rgb2lab(request.image).astype(np.float32)
        count = int(labels.max()) + 1
        features = np.zeros((count, 6), dtype=np.float32)
        yy, xx = np.indices(labels.shape)
        for segment in range(count):
            selected = labels == segment
            if not selected.any():
                raise RuntimeError(f"empty SLIC segment {segment}")
            features[segment, :3] = lab[selected].mean(0)
            features[segment, 3:] = lab[selected].std(0)
        scale = np.asarray([100, 128, 128, 50, 50, 50], dtype=np.float32)
        features /= scale
        positive = self._prompt_segments(labels, request.positive_points)
        negative = self._prompt_segments(labels, request.negative_points) if len(request.negative_points) else np.empty(0, np.int64)
        positive_distance = ((features[:, None] - features[positive][None]) ** 2).mean(2).min(1)
        if len(negative):
            negative_distance = ((features[:, None] - features[negative][None]) ** 2).mean(2).min(1)
            score = negative_distance - positive_distance
        else:
            score = -positive_distance
        probability_by_segment = 1.0 / (1.0 + np.exp(-self.temperature * score))
        probability = probability_by_segment[labels].astype(np.float32)
        return Prediction(
            probability=probability,
            binary_mask=(probability >= self.threshold),
            candidates={
                "selection_rule": "single_deterministic_mask",
                "positive_segments": positive.tolist(), "negative_segments": negative.tolist(),
                "num_segments": count,
            },
        )

