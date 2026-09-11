# Oracle Script Recognition Pipeline

[![Tests](https://github.com/jingsiihu-ai/oracle-script-recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/jingsiihu-ai/oracle-script-recognition/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean-room, dataset-independent reference pipeline for detecting, cropping, normalizing, and recognizing characters in oracle-bone imagery.

The project packages the reusable engineering components around a detector/recognizer stack: robust image preprocessing, IoU and non-maximum suppression, crop normalization, prototype-based recognition, and evaluation metrics. Detector backends such as YOLO or Faster R-CNN can be connected through a small `Detection` interface.

## Pipeline


## What is included

- percentile contrast normalization and Otsu binarization;
- bounding-box validation, pairwise IoU, and class-aware NMS;
- padded crop extraction and fixed-size glyph normalization;
- a transparent nearest-prototype recognizer for end-to-end testing;
- detection precision/recall and recognition accuracy utilities;
- synthetic demo, unit tests, and continuous integration.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/synthetic_demo.py
pytest
```

## Connecting a learned detector

Convert each model output to:

```python
Detection(box=(x1, y1, x2, y2), score=0.93, label="oracle_character")
```

Then call `non_max_suppression` and `extract_crops`. The rest of the pipeline is independent of the training framework.

## Data and result policy

No restricted scans, annotations, trained weights, or copied project code are included. The demo uses programmatically generated glyphs. This repository does not claim to reproduce previously reported research metrics; results should be published only with a documented dataset split, configuration, and evaluation artifact.

See [docs/method.md](docs/method.md) for extension points and references.

