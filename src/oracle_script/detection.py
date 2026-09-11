"""Framework-neutral bounding-box utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detector output in pixel-space xyxy format."""

    box: tuple[float, float, float, float]
    score: float
    label: str = "oracle_character"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Intersection over union of two xyxy boxes."""

    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def non_max_suppression(
    detections: list[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """Greedy, class-aware non-maximum suppression."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        overlaps = (
            candidate.label == selected.label
            and box_iou(candidate.box, selected.box) > iou_threshold
            for selected in kept
        )
        if not any(overlaps):
            kept.append(candidate)
    return kept

