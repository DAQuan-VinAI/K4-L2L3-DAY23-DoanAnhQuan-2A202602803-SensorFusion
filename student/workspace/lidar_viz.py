"""LiDAR range-image channel extraction and visualization scaling.

Provided Part A — read to understand ingest; do not edit for fusion submission.
"""

# vi: Phần cung cấp sẵn — đọc hiểu range image; không cần sửa để nộp fusion.

from __future__ import annotations

import numpy as np


def _normalize_range_channel(range_channel: np.ndarray) -> np.ndarray:
    """Map the range channel to unsigned 8-bit values using full dynamic range."""
    channel = np.array(range_channel, dtype=np.float64)
    channel[channel < 0] = 0.0
    span = np.amax(channel) - np.amin(channel)
    if span <= 0:
        return np.zeros(channel.shape, dtype=np.uint8)
    scaled = channel * 255.0 / span
    return scaled.astype(np.uint8)


def _normalize_intensity_channel(intensity_channel: np.ndarray) -> np.ndarray:
    """Map intensity to uint8 using 1st–99th percentile scaling to reduce outliers."""
    channel = np.array(intensity_channel, dtype=np.float64)
    channel[channel < 0] = 0.0
    p_low, p_high = np.percentile(channel, 1), np.percentile(channel, 99)
    if p_high <= p_low:
        return np.zeros(channel.shape, dtype=np.uint8)
    clipped = np.clip(channel, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low) * 255.0
    return scaled.astype(np.uint8)


def range_image_channels(range_image: np.ndarray) -> np.ndarray:
    """Extract and normalize range and intensity channels into a vertical uint8 stack."""
    ri = np.asarray(range_image)
    if ri.ndim != 3 or ri.shape[2] < 2:
        raise ValueError("range_image must have shape (H, W, C) with C >= 2")
    img_range = _normalize_range_channel(ri[:, :, 0])
    img_intensity = _normalize_intensity_channel(ri[:, :, 1])
    stacked = np.vstack((img_range, img_intensity))
    return stacked.astype(np.uint8)
