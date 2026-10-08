"""Regressions for the supplied detection and BEV baseline."""

import numpy as np
import pytest
from easydict import EasyDict
from shapely.geometry import Polygon
import torch


@pytest.mark.parametrize("yaw", [0.0, np.pi / 6, np.pi / 2, -np.pi / 3])
def test_box_area_matches_dimensions(workspace_modules, yaw):
    metrics = workspace_modules["detection_metrics"]
    box = metrics.box_corners(10.0, -3.0, 2.0, 4.0, yaw)
    assert Polygon(box).is_valid
    assert Polygon(box).area == pytest.approx(8.0)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 6, np.pi / 2])
@pytest.mark.parametrize("along_length, expected", [(True, 7 / 9), (False, 0.6)])
def test_offset_box_iou(workspace_modules, yaw, along_length, expected):
    metrics = workspace_modules["detection_metrics"]
    # Waymo defines length along the heading; width is perpendicular to it.
    if along_length:
        dx, dy = 0.5 * np.cos(yaw), 0.5 * np.sin(yaw)
    else:
        dx, dy = -0.5 * np.sin(yaw), 0.5 * np.cos(yaw)
    a = metrics.box_corners(0.0, 0.0, 2.0, 4.0, yaw)
    b = metrics.box_corners(dx, dy, 2.0, 4.0, yaw)
    assert metrics.rotated_iou(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "yaw, dx, dy, expected",
    [(0.0, 0.8, 0.0, 2 / 3), (0.0, 0.0, 0.8, 3 / 7),
     (np.pi / 2, 0.0, 0.8, 2 / 3)],
)
def test_waymo_heading_dimension_iou(workspace_modules, yaw, dx, dy, expected):
    metrics = workspace_modules["detection_metrics"]
    a = metrics.box_corners(0, 0, 2, 4, yaw)
    b = metrics.box_corners(dx, dy, 2, 4, yaw)
    assert metrics.rotated_iou(a, b) == pytest.approx(expected)


def test_sfa_heads_convert_to_waymo_boxes(workspace_modules, monkeypatch):
    """Exercise real decode/post-processing with controlled model-head tensors."""
    pipeline = workspace_modules["detection_pipeline"]
    configs = pipeline.load_fpn_resnet_config()
    configs.k = 7
    configs.bev_height = configs.bev_width = 24
    configs.lim_x = [10, 60]
    configs.lim_y = [5, 55]
    recorded_k = []
    real_decode = pipeline.decode

    def record_decode(*args, K):
        recorded_k.append(K)
        return real_decode(*args, K=K)

    def model_heads(_input):
        heatmap = torch.full((1, 3, 6, 6), -20.0)
        heatmap[0, 1, 2, 3] = 10.0
        direction = torch.zeros((1, 2, 6, 6))
        direction[:, 0] = np.sin(0.6)
        direction[:, 1] = np.cos(0.6)
        dimensions = torch.zeros((1, 3, 6, 6))
        dimensions[:, 0] = 1.8
        dimensions[:, 1] = 2.0
        dimensions[:, 2] = 4.0
        return {
            "hm_cen": heatmap,
            "cen_offset": torch.zeros((1, 2, 6, 6)),
            "direction": direction,
            "z_coor": torch.full((1, 1, 6, 6), 0.2),
            "dim": dimensions,
        }

    monkeypatch.setattr(pipeline, "decode", record_decode)
    detections = pipeline.detect_objects_from_bev(
        torch.zeros((1, 3, 24, 24)), model_heads, configs
    )
    assert recorded_k == [7]
    assert len(detections) == 1
    np.testing.assert_allclose(
        detections[0], [1, 10 + 10 / 24 * 50, 5 + 14 / 24 * 50,
                        0.1, 1.8, 2, 4, -0.6], atol=1e-6,
    )


def test_topmost_point_uses_maximum_height(workspace_modules):
    bev = workspace_modules["bev_mapping"]
    pcl = np.array([[1, 2, 0.25, 0.1], [1, 2, 0.875, 0.9], [1, 2, 0.5, 0.4]])
    top, counts = bev.topmost_points_per_cell(pcl)
    np.testing.assert_array_equal(top, pcl[[1]])
    np.testing.assert_array_equal(counts, [3])


def test_far_boundary_is_inside_bev(workspace_modules):
    bev = workspace_modules["bev_mapping"]
    configs = EasyDict(
        lim_x=[0, 50], lim_y=[-25, 25], lim_z=[-1, 3],
        bev_width=8, bev_height=8,
    )
    pcl = np.array([[50.0, 25.0, 1.0, 0.7]])
    discrete = bev.discretize_xy(bev.filter_pcl_for_bev(pcl, configs), configs)
    np.testing.assert_array_equal(discrete[0, :2], [7, 7])
    maps = bev.bev_maps_from_pcl(pcl, configs)
    assert maps.shape == (3, 8, 8)
    assert np.isfinite(maps).all()
    assert maps[1, 7, 7] == pytest.approx(0.5)
    assert maps[2, 7, 7] > 0
