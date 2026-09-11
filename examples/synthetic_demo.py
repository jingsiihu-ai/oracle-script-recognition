"""Run the pipeline on programmatically generated line glyphs."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from oracle_script import (
    Detection,
    PrototypeRecognizer,
    extract_crops,
    non_max_suppression,
    otsu_binarize,
)


def make_canvas() -> Image.Image:
    image = Image.new("L", (96, 48), color=235)
    draw = ImageDraw.Draw(image)
    draw.line((12, 10, 12, 36), fill=25, width=4)
    draw.line((8, 23, 28, 23), fill=25, width=4)
    draw.rectangle((56, 10, 78, 35), outline=25, width=4)
    noise = np.random.default_rng(4).normal(0, 4, (48, 96))
    return Image.fromarray(np.clip(np.asarray(image) + noise, 0, 255).astype(np.uint8))


def main() -> None:
    image = make_canvas()
    raw = [
        Detection((6, 7, 31, 39), 0.96),
        Detection((7, 8, 30, 38), 0.72),
        Detection((53, 7, 81, 39), 0.94),
    ]
    detections = non_max_suppression(raw)
    crops = extract_crops(image, detections)
    binary = otsu_binarize(image)

    recognizer = PrototypeRecognizer().fit(crops, ["cross", "box"])
    predictions = recognizer.predict(crops)
    print(f"foreground pixels: {int(binary.sum())}")
    print(f"detections after NMS: {len(detections)}")
    print(f"predicted labels: {predictions}")


if __name__ == "__main__":
    main()

