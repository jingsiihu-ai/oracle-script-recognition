# Method notes

## Design

The repository separates detector-specific inference from the post-processing and recognition pipeline. A learned detector only needs to emit `Detection` objects. This keeps NMS, crop normalization, evaluation, and recognizer experiments testable without GPU weights or a restricted dataset.

The included nearest-prototype recognizer is deliberately transparent. It is useful for integration tests and small few-shot experiments, while a production experiment can replace it with a CNN or vision-transformer adapter.

## Evaluation checklist

1. Version the image source, annotation policy, and train/validation/test split.
2. Report detector AP at declared IoU thresholds rather than an unlabeled aggregate.
3. Evaluate recognition on ground-truth crops and predicted crops separately.
4. Preserve class-frequency statistics and include macro metrics for long-tailed scripts.
5. Publish seeds, thresholds, preprocessing settings, and failure examples.

## References

- Ren et al., [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS 2015.
- Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020.
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

