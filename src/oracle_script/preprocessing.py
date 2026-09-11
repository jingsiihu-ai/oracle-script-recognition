"""Image preprocessing and crop extraction."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .detection import Detection


def _as_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float32)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise ValueError("image must be two-dimensional or RGB")
    return array.astype(np.float32)


def contrast_stretch(
    image: Image.Image | np.ndarray, lower: float = 2.0, upper: float = 98.0
) -> np.ndarray:
    """Robustly map grayscale intensities to uint8 [0, 255]."""

    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    array = _as_grayscale(image)
    low, high = np.percentile(array, [lower, upper])
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def otsu_binarize(image: Image.Image | np.ndarray, invert: bool = True) -> np.ndarray:
    """Binarize an image with Otsu's between-class variance criterion."""

    array = contrast_stretch(image)
    histogram = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    probability = histogram / max(histogram.sum(), 1.0)
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * np.arange(256))
    total_mean = mean[-1]
    denominator = omega * (1.0 - omega)
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    variance[valid] = (
        (total_mean * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    )
    threshold = int(np.argmax(variance))
    foreground = array <= threshold if invert else array > threshold
    return foreground.astype(np.uint8)


def extract_crops(
    image: Image.Image | np.ndarray,
    detections: list[Detection],
    padding: int = 2,
) -> list[np.ndarray]:
    """Clip padded boxes to image bounds and return grayscale crops."""

    if padding < 0:
        raise ValueError("padding must be non-negative")
    array = _as_grayscale(image).astype(np.uint8)
    height, width = array.shape
    crops: list[np.ndarray] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        left = max(0, int(np.floor(x1)) - padding)
        top = max(0, int(np.floor(y1)) - padding)
        right = min(width, int(np.ceil(x2)) + padding)
        bottom = min(height, int(np.ceil(y2)) + padding)
        crops.append(array[top:bottom, left:right].copy())
    return crops


def normalize_glyph(crop: Image.Image | np.ndarray, size: int = 32) -> np.ndarray:
    """Resize a grayscale crop onto a square canvas while preserving aspect ratio."""

    if size < 4:
        raise ValueError("size must be at least 4")
    array = contrast_stretch(crop)
    image = Image.fromarray(array, mode="L")
    scale = (size - 4) / max(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("L", (size, size), color=255)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return np.asarray(canvas, dtype=np.float32) / 255.0

