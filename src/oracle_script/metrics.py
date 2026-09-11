"""Small evaluation utilities with explicit matching behavior."""

from __future__ import annotations

from collections.abc import Sequence

from .detection import Detection, box_iou


def detection_precision_recall(
    predictions: Sequence[Detection],
    targets: Sequence[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute precision and recall using greedy score-ordered matching."""

    matched: set[int] = set()
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (box_iou(prediction.box, target), index)
            for index, target in enumerate(targets)
            if index not in matched
        ]
        if candidates:
            overlap, target_index = max(candidates)
            if overlap >= iou_threshold:
                matched.add(target_index)
                true_positive += 1
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(targets) if targets else 0.0
    return precision, recall


def recognition_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not targets:
        return 0.0
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)

