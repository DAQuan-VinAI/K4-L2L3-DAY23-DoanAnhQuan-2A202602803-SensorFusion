from types import SimpleNamespace

import numpy as np


def test_rotated_iou_identical(workspace_modules):
    dm = workspace_modules["detection_metrics"]
    box = dm.box_corners(10.0, 0.0, 2.0, 4.0, 0.0)
    iou = dm.rotated_iou(box, box)
    assert iou > 0.99


def test_precision_recall_counts(workspace_modules):
    dm = workspace_modules["detection_metrics"]
    counts = dm.precision_recall_counts([True, True], 1, 2)
    p, r = dm.precision_recall_from_counts(counts)
    assert 0 <= p <= 1
    assert 0 <= r <= 1
