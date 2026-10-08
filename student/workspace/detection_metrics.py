"""Object-detection evaluation helpers (IoU, precision/recall counts).

Provided Part D — read metrics baseline; do not edit for fusion submission.
"""

# vi: Phần cung cấp sẵn — IoU/P/R baseline; không cần sửa để nộp fusion.

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from shapely.geometry import Polygon


def box_corners(
    x: float, y: float, w: float, l: float, yaw: float
) -> list[tuple[float, float]]:
    """Return box corners using Waymo's length-along-heading convention.

    Args:
        x: Vehicle-frame box centre x coordinate in metres.
        y: Vehicle-frame box centre y coordinate in metres.
        w: Width perpendicular to the heading in metres.
        l: Length along the heading in metres.
        yaw: Heading from the vehicle-frame positive x axis, in radians.

    Returns:
        Corners in front-left, rear-left, rear-right, front-right order.
    """
    heading = np.array([np.cos(yaw), np.sin(yaw)]) * (l / 2)
    lateral = np.array([-np.sin(yaw), np.cos(yaw)]) * (w / 2)
    centre = np.array([x, y])
    return [
        tuple(centre + heading + lateral),
        tuple(centre - heading + lateral),
        tuple(centre - heading - lateral),
        tuple(centre + heading - lateral),
    ]


def rotated_iou(
    box_a: Sequence[tuple[float, float]], box_b: Sequence[tuple[float, float]]
) -> float:
    """Compute intersection-over-union for two rotated rectangles given as corner lists."""
    poly_a = Polygon(box_a)
    poly_b = Polygon(box_b)
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return float(intersection / union)


def center_distance(label_box: Any, detection: Sequence[Any]) -> tuple[float, float, float]:
    """Return absolute center offsets (dx, dy, dz) between label and detection."""
    _id, x, y, z, _h, _w, _l, _yaw = detection
    dist_x = float(label_box.center_x - x)
    dist_y = float(label_box.center_y - y)
    dist_z = float(label_box.center_z - z)
    return dist_x, dist_y, dist_z


def match_label_to_detections(
    label: Any, detections: Sequence[Any], min_iou: float = 0.5
) -> list[list[float]]:
    """Find detection matches for one label; returns list of [iou, dx, dy, dz]."""
    box_label = box_corners(
        label.box.center_x,
        label.box.center_y,
        label.box.width,
        label.box.length,
        label.box.heading,
    )
    matches = []
    for detection in detections:
        _id, x, y, z, _h, w, l, yaw = detection
        box_det = box_corners(x, y, w, l, yaw)
        iou = rotated_iou(box_label, box_det)
        if iou > min_iou:
            dx, dy, dz = center_distance(label.box, detection)
            matches.append([iou, dx, dy, dz])
    return matches


def precision_recall_counts(
    labels_valid: Sequence[bool] | np.ndarray,
    true_positives: int,
    num_detections: int,
) -> dict[str, int]:
    """Compute all positives, false negatives, and false positives for one frame."""
    all_positives = int(np.sum(labels_valid))
    false_negatives = all_positives - true_positives
    false_positives = num_detections - true_positives
    return {
        "all_positives": all_positives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def precision_recall_from_counts(counts: dict[str, int]) -> tuple[float, float]:
    """Compute precision and recall from aggregated count dict."""
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
