"""Reusable components for oracle script detection and recognition."""

from .detection import Detection, box_iou, non_max_suppression
from .metrics import detection_precision_recall, recognition_accuracy
from .preprocessing import contrast_stretch, extract_crops, normalize_glyph, otsu_binarize
from .recognition import PrototypeRecognizer

__all__ = [
    "Detection",
    "PrototypeRecognizer",
    "box_iou",
    "contrast_stretch",
    "detection_precision_recall",
    "extract_crops",
    "non_max_suppression",
    "normalize_glyph",
    "otsu_binarize",
    "recognition_accuracy",
]

