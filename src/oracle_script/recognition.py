"""Transparent recognition baseline used to exercise the full pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .preprocessing import normalize_glyph


class PrototypeRecognizer:
    """Nearest-class-centroid recognizer over normalized glyph pixels."""

    def __init__(self, image_size: int = 32) -> None:
        self.image_size = image_size
        self.labels: list[str] = []
        self.prototypes: np.ndarray | None = None

    def fit(self, crops: Sequence[np.ndarray], labels: Sequence[str]) -> "PrototypeRecognizer":
        if len(crops) != len(labels) or not crops:
            raise ValueError("crops and labels must be non-empty and have equal length")
        features = np.stack(
            [normalize_glyph(crop, self.image_size).ravel() for crop in crops]
        )
        self.labels = sorted(set(labels))
        self.prototypes = np.stack(
            [features[np.asarray(labels) == label].mean(axis=0) for label in self.labels]
        )
        return self

    def predict(self, crops: Sequence[np.ndarray]) -> list[str]:
        if self.prototypes is None:
            raise RuntimeError("fit must be called before predict")
        features = np.stack(
            [normalize_glyph(crop, self.image_size).ravel() for crop in crops]
        )
        distance = ((features[:, None, :] - self.prototypes[None, :, :]) ** 2).mean(-1)
        return [self.labels[index] for index in distance.argmin(axis=1)]

