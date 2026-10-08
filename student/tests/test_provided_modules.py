"""Tests for provided Part A–D modules (student pack baseline)."""

def test_student_bev_maps(workspace_modules):
    from easydict import EasyDict as edict

    bev = workspace_modules["bev_mapping"]
    configs = edict(
        lim_x=[0, 50],
        lim_y=[-25, 25],
        lim_z=[-1, 3],
        bev_width=608,
        bev_height=608,
    )
    import numpy as np

    pcl = np.random.rand(100, 4).astype(np.float32)
    pcl[:, 0] *= 40
    pcl[:, 1] = pcl[:, 1] * 50 - 25
    pcl[:, 2] = pcl[:, 2] * 2
    maps = bev.bev_maps_from_pcl(pcl, configs)
    assert maps.shape == (3, 608, 608)


def test_student_detection_metrics_iou(workspace_modules):
    dm = workspace_modules["detection_metrics"]
    box = dm.box_corners(10.0, 0.0, 2.0, 4.0, 0.0)
    assert dm.rotated_iou(box, box) > 0.99
