import pytest


def test_rotated_iou_identical(workspace_modules):
    dm = workspace_modules["detection_metrics"]
    box = dm.box_corners(10.0, 0.0, 2.0, 4.0, 0.0)
    iou = dm.rotated_iou(box, box)
    assert iou > 0.99


def test_precision_recall_counts(workspace_modules):
    dm = workspace_modules["detection_metrics"]
    counts = dm.precision_recall_counts([True, True], 1, 2)
    p, r = dm.precision_recall_from_counts(counts)
    assert counts == {
        "all_positives": 2,
        "true_positives": 1,
        "false_negatives": 1,
        "false_positives": 1,
    }
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
