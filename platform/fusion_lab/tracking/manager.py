"""Track storage and lidar-only lifecycle coordination for the lab EKF."""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab import tracking_params as params


def _vehicle_yaw(meas: Any) -> float:
    """Rotate a heading vector, retaining the sign of its vehicle-frame angle."""
    heading = np.array([np.cos(meas.yaw), np.sin(meas.yaw), 0.0])
    direction = np.asarray(meas.sensor.sens_to_veh)[:3, :3] @ heading
    return float(np.arctan2(direction[1], direction[0]))


class Track:
    """Store a student's initialized EKF state and vehicle-frame box attributes.

    Args:
        meas: Lidar observation that starts the track.
        track_id: Identifier assigned by the manager.
        track_mgmt: Student module supplying initialization and lifecycle rules.
    """

    def __init__(self, meas: Any, track_id: int, track_mgmt: Any) -> None:
        initial = track_mgmt.init_track_state_from_meas(meas)
        self.x, self.P = initial["x"], initial["P"]
        self.score, self.state = initial["score"], initial["state"]
        self.id = track_id
        self.height, self.width, self.length = meas.height, meas.width, meas.length
        self.yaw = _vehicle_yaw(meas)
        self.set_t(meas.t)

    def set_t(self, t: float) -> None:
        """Set nonnegative dataset-relative time in seconds.

        Args:
            t: Time relative to dataset frame zero.

        Raises:
            ValueError: If time is negative or nonfinite.
        """
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"Track timestamp must be finite and nonnegative: {t}")
        self.t = float(t)

    def update_attributes(self, meas: Any) -> None:
        """Smooth lidar dimensions and replace heading with its signed rotation.

        Args:
            meas: Associated observation; camera observations leave dimensions alone.
        """
        if meas.sensor.name != "lidar":
            return
        previous = np.array([self.height, self.width, self.length])
        observed = np.array([meas.height, meas.width, meas.length])
        dimensions = previous + params.weight_dim * (observed - previous)
        self.height, self.width, self.length = map(float, dimensions)
        self.yaw = _vehicle_yaw(meas)


class TrackManager:
    """Coordinate lidar hits, visible misses, deletions, and births.

    A camera pass only refines EKF state; it never scores, deletes, or creates
    tracks. Call ``manage_tracks`` even when a lidar frame has no measurements.

    Args:
        track_mgmt: Student module supplying score and deletion decisions.
    """

    def __init__(self, track_mgmt: Any) -> None:
        self._tm = track_mgmt
        self.track_list: list[Track] = []
        self.result_list: list[Any] = []
        self.last_id = -1

    def _record_observation(self, track: Track, associated: bool) -> None:
        result = self._tm.update_track_score(
            {"score": track.score, "state": track.state}, associated=associated
        )
        track.score, track.state = result["score"], result["state"]

    def handle_updated_track(self, track: Track, sensor: Any) -> None:
        """Record one lidar hit after an EKF update.

        Args:
            track: Associated track.
            sensor: Sensor for this pass; camera hits do not affect existence.
        """
        if sensor.name == "lidar":
            self._record_observation(track, associated=True)

    def manage_tracks(
        self,
        unassigned_tracks: Sequence[Track],
        unassigned_meas: Sequence[Any],
        sensor: Any,
    ) -> None:
        """Finish a sensor pass, including lidar frames with no detections.

        Args:
            unassigned_tracks: Tracks without an associated observation this pass.
            unassigned_meas: Unmatched observations eligible for lidar births.
            sensor: Explicit pass sensor, independent of observation-list contents.
        """
        if sensor.name != "lidar":
            return
        for track in unassigned_tracks:
            if sensor.in_fov(track.x):
                self._record_observation(track, associated=False)
        survivors = []
        for track in self.track_list:
            lifecycle = {"score": track.score, "state": track.state, "P": track.P}
            if not self._tm.should_delete_track(lifecycle):
                survivors.append(track)
        self.track_list = survivors
        for meas in unassigned_meas:
            if meas.sensor.name == "lidar":
                self.last_id += 1
                self.track_list.append(Track(meas, self.last_id, self._tm))
