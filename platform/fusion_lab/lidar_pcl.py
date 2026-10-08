"""Convert Waymo lidar range images using the Apache-licensed reader."""

from __future__ import annotations

import numpy as np
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils


def pcl_from_range_image(frame: dataset_pb2.Frame, lidar_name: int) -> np.ndarray:
    """Return valid lidar points in vehicle coordinates with their intensities.

    Args:
        frame: Waymo frame containing lidar returns and calibration.
        lidar_name: LaserName identifier of the sensor to project.

    Returns:
        An (N, 4) array of x, y, z and intensity, retaining only positive ranges.
    """
    laser = utils.get(frame.lasers, lidar_name)
    calibration = utils.get(frame.context.laser_calibrations, lidar_name)
    ri, camera_projection, ri_pose = utils.parse_range_image_and_camera_projection(laser)
    points, attributes = utils.project_to_pointcloud(
        frame, ri, camera_projection, ri_pose, calibration
    )
    return np.column_stack((points, attributes[:, 1]))
