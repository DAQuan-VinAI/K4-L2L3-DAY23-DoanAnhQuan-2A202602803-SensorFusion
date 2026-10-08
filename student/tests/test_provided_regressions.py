"""Regressions for the supplied detection and BEV baseline."""

import numpy as np
import pytest
from easydict import EasyDict
from shapely.geometry import Polygon


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
    # The supplied corner convention has length along (-sin(yaw), cos(yaw)).
    if along_length:
        dx, dy = -0.5 * np.sin(yaw), 0.5 * np.cos(yaw)
    else:
        dx, dy = 0.5 * np.cos(yaw), 0.5 * np.sin(yaw)
    a = metrics.box_corners(0.0, 0.0, 2.0, 4.0, yaw)
    b = metrics.box_corners(dx, dy, 2.0, 4.0, yaw)
    assert metrics.rotated_iou(a, b) == pytest.approx(expected)


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
