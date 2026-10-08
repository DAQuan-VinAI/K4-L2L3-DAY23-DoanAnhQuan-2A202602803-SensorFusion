"""Sensor and measurement objects (Jacobian H provided; h(x) from student camera_fusion)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from fusion_lab import tracking_params as params

Matrix = np.matrix | np.ndarray


class Sensor:
    """LiDAR or front camera sensor with extrinsics/intrinsics."""

    def __init__(self, name: str, calib: Any, camera_fusion: Any) -> None:
        self.name = name
        self._cam = camera_fusion
        if name == "lidar":
            self.dim_meas = 3
            self.sens_to_veh = np.asmatrix(np.identity(4))
            self.fov = [-np.pi / 2, np.pi / 2]
        elif name == "camera":
            self.dim_meas = 2
            self.sens_to_veh = np.asmatrix(calib.extrinsic.transform).reshape(4, 4)
            self.f_i = calib.intrinsic[0]
            self.f_j = calib.intrinsic[1]
            self.c_i = calib.intrinsic[2]
            self.c_j = calib.intrinsic[3]
            self.fov = [-0.35, 0.35]
        else:
            raise ValueError(name)
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh)

    def in_fov(self, x: Matrix) -> bool:
        """Return True if state ``x`` lies inside this sensor's field of view."""
        return self._cam.is_in_field_of_view(x, self)

    def get_hx(self, x: Matrix) -> Matrix:
        """Return predicted measurement h(x) in sensor coordinates."""
        if self.name == "lidar":
            pos_veh = np.ones((4, 1))
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens * pos_veh
            return pos_sens[0:3]
        return self._cam.camera_measurement_prediction(x, self)

    def get_H(self, x: Matrix) -> Matrix:
        """Return measurement Jacobian H = dh/dx at state ``x``."""
        H = np.asmatrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3]
        T = self.veh_to_sens[0:3, 3]
        if self.name == "lidar":
            H[0:3, 0:3] = R
        elif self.name == "camera":
            px, py, pz = float(x[0, 0]), float(x[1, 0]), float(x[2, 0])
            Rf = np.asarray(R, dtype=np.float64)
            Tf = np.asarray(T, dtype=np.float64).reshape(3)
            denom = float(Rf[0, 0] * px + Rf[0, 1] * py + Rf[0, 2] * pz + Tf[0])
            if denom == 0:
                raise ValueError(
                    "Camera Jacobian undefined: projected x coordinate is zero"
                )
            H[0, 0] = float(
                self.f_i
                * (
                    -Rf[1, 0] / denom
                    + Rf[0, 0]
                    * (Rf[1, 0] * px + Rf[1, 1] * py + Rf[1, 2] * pz + Tf[1])
                    / (denom**2)
                )
            )
            H[1, 0] = float(
                self.f_j
                * (
                    -Rf[2, 0] / denom
                    + Rf[0, 0]
                    * (Rf[2, 0] * px + Rf[2, 1] * py + Rf[2, 2] * pz + Tf[2])
                    / (denom**2)
                )
            )
            H[0, 1] = float(
                self.f_i
                * (
                    -Rf[1, 1] / denom
                    + Rf[0, 1]
                    * (Rf[1, 0] * px + Rf[1, 1] * py + Rf[1, 2] * pz + Tf[1])
                    / (denom**2)
                )
            )
            H[1, 1] = float(
                self.f_j
                * (
                    -Rf[2, 1] / denom
                    + Rf[0, 1]
                    * (Rf[2, 0] * px + Rf[2, 1] * py + Rf[2, 2] * pz + Tf[2])
                    / (denom**2)
                )
            )
            H[0, 2] = float(
                self.f_i
                * (
                    -Rf[1, 2] / denom
                    + Rf[0, 2]
                    * (Rf[1, 0] * px + Rf[1, 1] * py + Rf[1, 2] * pz + Tf[1])
                    / (denom**2)
                )
            )
            H[1, 2] = float(
                self.f_j
                * (
                    -Rf[2, 2] / denom
                    + Rf[0, 2]
                    * (Rf[2, 0] * px + Rf[2, 1] * py + Rf[2, 2] * pz + Tf[2])
                    / (denom**2)
                )
            )
        return H

    def generate_measurement(
        self, num_frame: int, z: Sequence[float], meas_list: list[Any]
    ) -> list[Any]:
        """Append a ``Measurement`` built from raw detection ``z`` to ``meas_list``.

        Args:
            num_frame: Waymo frame index (1-based in the lab loop).
            z: Lidar box vector or camera label coordinates.
            meas_list: List mutated in place.

        Returns:
            The same ``meas_list`` reference for chaining.
        """
        if self.name == "lidar":
            meas_list.append(Measurement(num_frame, z, self))
        elif self.name == "camera":
            payload = self._cam.build_camera_measurement(z, self)
            z_mat = payload["z"]
            meas_list.append(
                Measurement(
                    num_frame,
                    [float(z_mat[0, 0]), float(z_mat[1, 0])],
                    self,
                    z_mat=z_mat,
                    R=payload["R"],
                )
            )
        return meas_list


class Measurement:
    """Single sensor measurement with z, R, and optional box attributes."""

    def __init__(
        self,
        num_frame: int,
        z: Sequence[float],
        sensor: Sensor,
        z_mat: Optional[Matrix] = None,
        R: Optional[Matrix] = None,
    ) -> None:
        """Build a measurement at frame ``num_frame`` from detection ``z``.

        Args:
            num_frame: Frame index used to compute timestamp ``t``.
            z: Raw detection vector (lidar 7-D box or camera u,v).
            sensor: Originating sensor.
            z_mat: Optional pre-built measurement matrix (camera).
            R: Optional measurement noise (camera); lidar uses params sigmas.
        """
        self.t = (num_frame - 1) * params.dt
        self.sensor = sensor
        if sensor.name == "lidar":
            self.z = np.asmatrix(np.array([[z[0]], [z[1]], [z[2]]], dtype=np.float64))
            self.R = np.asmatrix(
                [
                    [params.sigma_lidar_x**2, 0, 0],
                    [0, params.sigma_lidar_y**2, 0],
                    [0, 0, params.sigma_lidar_z**2],
                ]
            )
            self.height = z[3]
            self.width = z[4]
            self.length = z[5]
            self.yaw = z[6]
        else:
            self.z = z_mat if z_mat is not None else np.asmatrix(np.array([[z[0]], [z[1]]]))
            self.R = R
