"""Bird's-eye view rasterization from LiDAR point clouds.

Provided Part B — read to understand BEV; do not edit for fusion submission.
"""

# vi: Phần cung cấp sẵn — đọc hiểu BEV; không cần sửa để nộp fusion.

from __future__ import annotations

from typing import Any

import numpy as np


def bev_discretization(configs: Any) -> float:
    """Return metric cell size for BEV discretization along x and y."""
    return (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height


def discretize_xy(pcl: np.ndarray, configs: Any) -> np.ndarray:
    """Convert metric x/y coordinates into discrete BEV grid indices."""
    pcl_cpy = np.copy(pcl)
    cell = bev_discretization(configs)
    pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / cell))
    pcl_cpy[:, 1] = np.int_(np.floor(pcl_cpy[:, 1] / cell) + (configs.bev_width + 1) / 2)
    pcl_cpy[:, 1] = np.abs(pcl_cpy[:, 1])
    pcl_cpy[:, 0] = np.clip(pcl_cpy[:, 0], 0, configs.bev_height - 1)
    pcl_cpy[:, 1] = np.clip(pcl_cpy[:, 1], 0, configs.bev_width - 1)
    return pcl_cpy


def filter_pcl_for_bev(lidar_pcl: np.ndarray, configs: Any) -> np.ndarray:
    """Keep points inside the detection volume and shift z relative to lim_z[0]."""
    mask = np.where(
        (lidar_pcl[:, 0] >= configs.lim_x[0])
        & (lidar_pcl[:, 0] <= configs.lim_x[1])
        & (lidar_pcl[:, 1] >= configs.lim_y[0])
        & (lidar_pcl[:, 1] <= configs.lim_y[1])
        & (lidar_pcl[:, 2] >= configs.lim_z[0])
        & (lidar_pcl[:, 2] <= configs.lim_z[1])
    )
    filtered = lidar_pcl[mask].copy()
    filtered[:, 2] = filtered[:, 2] - configs.lim_z[0]
    return filtered


def topmost_points_per_cell(pcl_cpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return top-most point per BEV cell plus per-cell point counts."""
    sorted_idx = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    pcl_sorted = pcl_cpy[sorted_idx]
    _, indices, counts = np.unique(
        pcl_sorted[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    pcl_top = pcl_sorted[indices]
    return pcl_top, counts


def build_intensity_map(pcl_top: np.ndarray, configs: Any) -> np.ndarray:
    """Build a normalized intensity layer for the BEV grid."""
    intensity_map = np.zeros((configs.bev_height, configs.bev_width), dtype=np.float32)
    intensities = np.clip(pcl_top[:, 3], 0.0, 1.0)
    span = np.amax(intensities) - np.amin(intensities)
    if span > 0:
        norm = intensities / span
    else:
        norm = np.zeros_like(intensities)
    rows = np.int_(pcl_top[:, 0])
    cols = np.int_(pcl_top[:, 1])
    intensity_map[rows, cols] = norm
    return intensity_map


def build_height_map(pcl_top: np.ndarray, configs: Any) -> np.ndarray:
    """Build a normalized height layer for the BEV grid."""
    height_map = np.zeros((configs.bev_height, configs.bev_width), dtype=np.float32)
    z_span = float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    if z_span <= 0:
        return height_map
    norm_h = pcl_top[:, 2] / z_span
    rows = np.int_(pcl_top[:, 0])
    cols = np.int_(pcl_top[:, 1])
    height_map[rows, cols] = norm_h
    return height_map


def build_density_map(
    pcl_cpy: np.ndarray, pcl_top: np.ndarray, configs: Any
) -> np.ndarray:
    """Build a log-scaled point-density layer for the BEV grid."""
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1), dtype=np.float32)
    _, _, counts = np.unique(pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    rows = np.int_(pcl_top[:, 0])
    cols = np.int_(pcl_top[:, 1])
    density_map[rows, cols] = normalized
    return density_map[: configs.bev_height, : configs.bev_width]


def bev_maps_from_pcl(lidar_pcl: np.ndarray, configs: Any) -> np.ndarray:
    """Rasterize a point cloud into a 3xHxW BEV tensor (intensity, height, density)."""
    pcl = filter_pcl_for_bev(lidar_pcl, configs)
    pcl_cpy = discretize_xy(pcl, configs)
    pcl_top, _ = topmost_points_per_cell(pcl_cpy)
    intensity_map = build_intensity_map(pcl_top, configs)
    height_map = build_height_map(pcl_top, configs)
    density_map = build_density_map(pcl_cpy, pcl_top, configs)
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width), dtype=np.float32)
    bev_map[0, :, :] = intensity_map
    bev_map[1, :, :] = height_map
    bev_map[2, :, :] = density_map
    return bev_map
