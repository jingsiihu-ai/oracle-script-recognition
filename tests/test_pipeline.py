import numpy as np

from oracle_script import (
    Detection,
    PrototypeRecognizer,
    box_iou,
    contrast_stretch,
    detection_precision_recall,
    extract_crops,
    non_max_suppression,
    normalize_glyph,
    recognition_accuracy,
)


def test_iou_identity_and_disjoint() -> None:
    box = (0, 0, 10, 10)
    assert box_iou(box, box) == 1.0
    assert box_iou(box, (20, 20, 30, 30)) == 0.0


def test_nms_removes_lower_scored_overlap() -> None:
    detections = [
        Detection((0, 0, 10, 10), 0.9),
        Detection((1, 1, 11, 11), 0.6),
        Detection((20, 20, 30, 30), 0.8),
    ]
    kept = non_max_suppression(detections, iou_threshold=0.5)
    assert [item.score for item in kept] == [0.9, 0.8]


def test_preprocessing_and_crop_shapes() -> None:
    image = np.arange(400, dtype=np.float32).reshape(20, 20)
    stretched = contrast_stretch(image)
    crops = extract_crops(image, [Detection((2, 3, 10, 12), 0.9)], padding=1)
    assert stretched.dtype == np.uint8
    assert crops[0].shape == (11, 10)
    assert normalize_glyph(crops[0], size=24).shape == (24, 24)


def test_detection_metrics() -> None:
    predictions = [
        Detection((0, 0, 10, 10), 0.9),
        Detection((20, 20, 30, 30), 0.7),
    ]
    precision, recall = detection_precision_recall(predictions, [(0, 0, 10, 10)])
    assert precision == 0.5
    assert recall == 1.0


def test_prototype_recognizer_and_accuracy() -> None:
    vertical = np.full((12, 12), 255, dtype=np.uint8)
    vertical[:, 5:7] = 0
    horizontal = np.full((12, 12), 255, dtype=np.uint8)
    horizontal[5:7, :] = 0
    model = PrototypeRecognizer(image_size=16).fit(
        [vertical, horizontal], ["vertical", "horizontal"]
    )
    predictions = model.predict([vertical, horizontal])
    assert recognition_accuracy(predictions, ["vertical", "horizontal"]) == 1.0

