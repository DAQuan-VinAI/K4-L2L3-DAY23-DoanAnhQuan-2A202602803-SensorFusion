"""EKF filter delegating predict/update to student kalman module."""

from __future__ import annotations

from types import ModuleType
from typing import Any


class Filter:
    """Apply student Kalman functions to mutable platform tracks.

    Args:
        kalman_mod: Module from the selected student workspace.
    """

    def __init__(self, kalman_mod: ModuleType) -> None:
        self._k = kalman_mod

    def predict(self, track: Any) -> None:
        """Advance the stored state and covariance with the motion model.

        Args:
            track: Mutable object exposing ``x`` and ``P``.
        """
        x, P = self._k.ekf_predict(track.x, track.P)
        track.x = x
        track.P = P

    def update(self, track: Any, meas: Any) -> None:
        """Correct the stored estimate and update lidar box attributes.

        Args:
            track: Mutable estimate with ``update_attributes``.
            meas: Associated observation for the correction.
        """
        x, P = self._k.ekf_update(track.x, track.P, meas)
        track.x = x
        track.P = P
        track.update_attributes(meas)
