"""Track list lifecycle using student track_management helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab import tracking_params as params


class Track:
    """Single track with EKF state and lifecycle score."""

    def __init__(self, meas: Any, track_id: int, track_mgmt: Any) -> None:
        init = track_mgmt.init_track_state_from_meas(meas)
        self.x = init["x"]
        self.P = init["P"]
        self.state = init["state"]
        self.score = init["score"]
        self.id = track_id
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw = np.arccos(
            M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw)
        )
        self.t = meas.t

    def set_t(self, t: float) -> None:
        """Set the track timestamp."""
        self.t = t

    def update_attributes(self, meas: Any) -> None:
        """Smooth box dimensions and yaw from a lidar measurement."""
        if meas.sensor.name == "lidar":
            c = params.weight_dim
            self.width = c * meas.width + (1 - c) * self.width
            self.length = c * meas.length + (1 - c) * self.length
            self.height = c * meas.height + (1 - c) * self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(
                M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw)
            )


class TrackManager:
    """Initialize, score, and delete tracks."""

    def __init__(self, track_mgmt: Any) -> None:
        self._tm = track_mgmt
        self.track_list: list[Track] = []
        self.last_id = -1
        self.result_list: list[Any] = []

    def handle_updated_track(self, track: Track) -> None:
        """Apply associated-hit scoring via student track_management."""
        updated = self._tm.update_track_score(
            {"score": track.score, "state": track.state}, associated=True
        )
        track.score = updated["score"]
        track.state = updated["state"]

    def manage_tracks(
        self,
        unassigned_tracks: list[Any],
        unassigned_meas: list[Any],
        meas_list: list[Any],
    ) -> None:
        """Score misses, delete dead tracks, and birth tracks from lidar detections."""
        for track in unassigned_tracks:
            if meas_list and meas_list[0].sensor.in_fov(track.x):
                updated = self._tm.update_track_score(
                    {"score": track.score, "state": track.state}, associated=False
                )
                track.score = updated["score"]
                track.state = updated["state"]
        self.track_list = [
            t
            for t in self.track_list
            if not self._tm.should_delete_track(
                {"score": t.score, "state": t.state, "P": t.P}
            )
        ]
        for meas in unassigned_meas:
            if meas.sensor.name != "lidar":
                continue
            self.last_id += 1
            self.track_list.append(Track(meas, self.last_id, self._tm))
