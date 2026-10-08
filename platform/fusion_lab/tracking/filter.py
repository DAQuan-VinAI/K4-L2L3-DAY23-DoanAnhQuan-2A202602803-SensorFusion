"""EKF filter delegating predict/update to student kalman module."""

from __future__ import annotations

from types import ModuleType
from typing import Any

class Filter:
    """Wrap student kalman pure functions for track objects."""

    def __init__(self, kalman_mod: ModuleType) -> None:
        self._k = kalman_mod

    def predict(self, track: Any) -> None:
        """Run one EKF predict step on ``track`` using the student kalman module."""
        x, P = self._k.ekf_predict(track.x, track.P)
        track.x = x
        track.P = P

    def update(self, track: Any, meas: Any) -> None:
        """Run one EKF update on ``track`` with ``meas`` and refresh box attributes."""
        x, P = self._k.ekf_update(track.x, track.P, meas)
        track.x = x
        track.P = P
        track.update_attributes(meas)
