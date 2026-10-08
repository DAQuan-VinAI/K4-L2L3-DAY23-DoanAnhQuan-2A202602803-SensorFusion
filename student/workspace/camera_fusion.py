"""Camera field-of-view checks and pinhole measurement modeling.

Part G — implement ``# vi: TODO`` (camera path on README diagram).
Jacobian H is provided by platform; you implement h(x), FOV, and R here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Camera sensor with ``veh_to_sens``, ``fov`` (radians).

    Returns:
        True if horizontal angle in sensor frame lies in ``sensor.fov``.
    """
    # vi: TODO Part G — đưa x sang tọa độ sensor; angle = atan2(y_s, x_s);
    # vi: kiểm tra fov[0] <= angle <= fov[1].
    raise NotImplementedError("TODO: implement is_in_field_of_view")


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        NameError: If projection is undefined (e.g. point behind camera).
    """
    # vi: TODO Part G — pos_sens = veh_to_sens @ [x,y,z,1]; u,v pinhole từ cx,cy,cz;
    # vi: công thức lab: u = c_i - f_i * cy/cx, v = c_j - f_j * cz/cx; cx<=0 → lỗi.
    raise NotImplementedError("TODO: implement camera_measurement_prediction")


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    # vi: TODO Part G — z mat 2x1; R diag sigma_cam_i^2, sigma_cam_j^2 từ params.
    raise NotImplementedError("TODO: implement build_camera_measurement")


def register_camera_detection(
    num_frame: int,
    z: Sequence[float],
    sensor: Any,
    meas_list: list[Any],
) -> list[Any]:
    """Append a camera detection measurement dict to meas_list.

    Args:
        num_frame: Frame index from Waymo loop.
        z: Pixel measurement ``[u, v]``.
        sensor: Camera sensor.
        meas_list: List mutated in place.

    Returns:
        Updated ``meas_list``.
    """
    # vi: TODO Part G — t = (num_frame-1)*dt; append dict t, sensor, z, R.
    raise NotImplementedError("TODO: implement register_camera_detection")
