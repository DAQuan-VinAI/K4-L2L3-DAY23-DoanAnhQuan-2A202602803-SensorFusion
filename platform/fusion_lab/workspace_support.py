"""Helpers for student workspace modules running under fusion_lab."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from fusion_lab import tracking_params as _params
from fusion_lab.paths import PLATFORM_ROOT, THIRD_PARTY


def get_tracking_params() -> ModuleType:
    """Return the lab tracking parameter module (dt, noise, gating, lifecycle).

    Returns:
        Module with scalar constants used by EKF and track management.
    """
    return _params


def platform_root() -> Path:
    """Return the Day 23 platform root directory.

    Returns:
        Path to `platform`.
    """
    return PLATFORM_ROOT


def third_party_root() -> Path:
    """Return the platform third_party directory (Waymo reader, objdet models).

    Returns:
        Path to `platform/third_party`.
    """
    return THIRD_PARTY
