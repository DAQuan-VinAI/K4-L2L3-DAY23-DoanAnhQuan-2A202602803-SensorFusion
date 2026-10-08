"""Check projection geometry, association visibility, and lidar-only existence."""

from types import SimpleNamespace

import numpy as np
import pytest

from fusion_lab import tracking_params as params
from fusion_lab.tracking.manager import Track
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Measurement
from fusion_lab.tracking.sensors import MIN_CAMERA_DEPTH
from fusion_lab.tracking.sensors import Sensor


def _state(position):
    return np.asmatrix(np.r_[position, [0.0, 0.0, 0.0]]).T


def _projection(x, sensor):
    """Use the pinhole equations as an independent numerical test oracle."""
    transform = np.asarray(sensor.veh_to_sens)
    depth, left, up = transform[:3, :3] @ np.asarray(x).ravel()[:3] + transform[:3, 3]
    return np.asmatrix([
        sensor.c_i - sensor.f_i * left / depth,
        sensor.c_j - sensor.f_j * up / depth,
    ]).T


@pytest.fixture
def sensors():
    calibration = SimpleNamespace(
        intrinsic=[800.0, 900.0, 400.0, 300.0],
        width=1000,
        extrinsic=SimpleNamespace(transform=np.eye(4).ravel()),
    )
    camera_model = SimpleNamespace(
        is_in_field_of_view=lambda x, sensor: True,
        camera_measurement_prediction=_projection,
    )
    return (
        Sensor("lidar", None, camera_model),
        Sensor("camera", calibration, camera_model),
    )


def test_camera_fov_comes_from_calibration(sensors):
    _, camera = sensors
    np.testing.assert_allclose(camera.fov, np.arctan([-0.75, 0.5]))
    for angle, expected_pixel in zip(camera.fov, (camera.image_width, 0.0)):
        predicted = camera.get_hx(_state([10.0, 10.0 * np.tan(angle), 0.0]))
        assert float(predicted[0, 0]) == pytest.approx(expected_pixel, abs=1e-10)


@pytest.mark.parametrize("position", [
    [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [MIN_CAMERA_DEPTH, 0.0, 0.0], [1.0, np.nan, 0.0], [np.inf, 0.0, 0.0],
])
def test_camera_invalid_depth_or_coordinates(sensors, position):
    _, camera = sensors
    state = _state(position)
    assert not camera.in_fov(state)
    with pytest.raises(ValueError, match="positive depth"):
        camera.get_hx(state)
    with pytest.raises(ValueError, match="positive depth"):
        camera.get_H(state)


def test_camera_jacobian_matches_finite_differences(sensors):
    _, camera = sensors
    angle = 0.2
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0],
    ])
    camera.veh_to_sens[:3, :3] = rotation
    camera.veh_to_sens[:3, 3] = np.asmatrix([0.5, -0.2, 0.3]).T
    state = _state([12.0, -1.0, 0.4])
    jacobian = np.asarray(camera.get_H(state))
    numeric = np.empty((2, 6))
    step = 1e-5
    for coordinate in range(6):
        delta = np.zeros((6, 1))
        delta[coordinate, 0] = step
        numeric[:, coordinate] = np.asarray(
            (camera.get_hx(state + delta) - camera.get_hx(state - delta)) / (2 * step)
        ).ravel()
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-6, atol=1e-7)


def test_lidar_measurement_timestamp_and_covariance(sensors):
    lidar, _ = sensors
    observation = Measurement(0, [10.0, 1.0, 2.0, 1.5, 2.0, 4.0, -0.3], lidar)
    assert observation.t == 0.0
    assert Measurement(3, [10, 1, 2, 1.5, 2, 4, 0], lidar).t == pytest.approx(3 * params.dt)
    np.testing.assert_allclose(observation.z, [[10], [1], [2]])
    np.testing.assert_allclose(observation.R, np.diag([
        params.sigma_lidar_x**2, params.sigma_lidar_y**2, params.sigma_lidar_z**2,
    ]))
    with pytest.raises(ValueError, match="nonnegative"):
        Measurement(-1, [10, 1, 2, 1.5, 2, 4, 0], lidar)


class LifecycleProbe:
    """Record manager delegation without depending on unfinished exercises."""

    def __init__(self):
        self.hits = []
        self.deletions = []

    def init_track_state_from_meas(self, meas):
        return {
            "x": _state(np.asarray(meas.z).ravel()),
            "P": np.asmatrix(np.eye(6)), "score": 1 / params.window,
            "state": "initialized",
        }

    def update_track_score(self, track, associated):
        self.hits.append(associated)
        return {**track, "score": track["score"] + (1 if associated else -1) / params.window}

    def should_delete_track(self, track):
        self.deletions.append(track)
        return track["score"] <= 0


@pytest.mark.parametrize("yaw", [-0.7, 0.7])
def test_track_signed_yaw_and_rotated_heading(sensors, yaw):
    lidar, _ = sensors
    angle = 0.2
    lidar.sens_to_veh[:2, :2] = np.array([
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)],
    ])
    meas = Measurement(0, [10, 0, 0, 1, 2, 4, yaw], lidar)
    track = Track(meas, 4, LifecycleProbe())
    assert track.yaw == pytest.approx(yaw + angle)
    meas.yaw = -0.5
    track.update_attributes(meas)
    assert track.yaw == pytest.approx(-0.5 + angle)
    assert track.t == 0
    with pytest.raises(ValueError, match="nonnegative"):
        track.set_t(-0.1)


def test_empty_lidar_pass_scores_and_deletes(sensors):
    lidar, _ = sensors
    rules = LifecycleProbe()
    manager = TrackManager(rules)
    meas = Measurement(0, [10, 0, 0, 1, 2, 4, 0], lidar)
    manager.manage_tracks([], [meas], lidar)
    track = manager.track_list[0]
    manager.manage_tracks([track], [], lidar)
    assert rules.hits == [False]
    assert manager.track_list == []


def test_camera_pass_cannot_score_delete_or_birth(sensors):
    lidar, camera = sensors
    rules = LifecycleProbe()
    manager = TrackManager(rules)
    lidar_meas = Measurement(0, [10, 0, 0, 1, 2, 4, 0], lidar)
    manager.manage_tracks([], [lidar_meas], lidar)
    track = manager.track_list[0]
    track.P[0, 0] = 100
    camera_meas = Measurement(0, [400, 300], camera)
    rules.deletions.clear()
    manager.handle_updated_track(track, camera)
    manager.manage_tracks([track], [camera_meas], camera)
    assert rules.hits == []
    assert rules.deletions == []
    assert manager.track_list == [track]
    assert track.score == pytest.approx(1 / params.window)


def test_outside_lidar_fov_does_not_score_miss(sensors):
    lidar, _ = sensors
    rules = LifecycleProbe()
    manager = TrackManager(rules)
    lidar.in_fov = lambda x: False
    manager.manage_tracks([], [Measurement(0, [10, 0, 0, 1, 2, 4, 0], lidar)], lidar)
    manager.manage_tracks(manager.track_list[:], [], lidar)
    assert rules.hits == []
    assert len(manager.track_list) == 1


@pytest.mark.student_exercise
@pytest.mark.parametrize("position", [[0, 0, 0], [-1, 0, 0], [1, np.nan, 0]])
def test_student_camera_rejects_invalid_projection(workspace_modules, sensors, position):
    _, camera = sensors
    module = workspace_modules["camera_fusion"]
    assert not module.is_in_field_of_view(_state(position), camera)
    with pytest.raises(ValueError):
        module.camera_measurement_prediction(_state(position), camera)


@pytest.mark.student_exercise
@pytest.mark.parametrize("position, expected", [
    ([10, 0, 0], True), ([10, 6, 0], False), ([10, -8, 0], False),
])
def test_student_camera_visibility_and_front_projection(
    workspace_modules, sensors, position, expected
):
    _, camera = sensors
    module = workspace_modules["camera_fusion"]
    state = _state(position)
    assert module.is_in_field_of_view(state, camera) is expected
    np.testing.assert_allclose(module.camera_measurement_prediction(state, camera),
                               _projection(state, camera))


@pytest.mark.student_exercise
@pytest.mark.parametrize("camera_unmatched", [False, True])
def test_lidar_hits_confirm_despite_camera_passes(workspace_modules, sensors, camera_unmatched):
    lidar, camera = sensors
    manager = TrackManager(workspace_modules["track_management"])
    manager.manage_tracks([], [Measurement(0, [10, 0, 0, 1, 2, 4, 0], lidar)], lidar)
    track = manager.track_list[0]
    for frame in range(1, params.window):
        manager.handle_updated_track(track, lidar)
        manager.manage_tracks([], [], lidar)
        before = track.score
        observations = [Measurement(frame, [800, 300], camera)] if camera_unmatched else []
        manager.manage_tracks([track], observations, camera)
        assert track.score == before
        assert manager.track_list == [track]
    assert track.state == "confirmed"
    assert track.score == pytest.approx(1.0)
    manager.manage_tracks([track], [], lidar)
    assert track.state == "confirmed"
    assert manager.track_list == [track]


@pytest.mark.student_exercise
@pytest.mark.parametrize("state, score, variance, expected", [
    ("initialized", 0.0, 1.0, True), ("tentative", 0.2, 1.0, False),
    ("confirmed", params.delete_threshold, 1.0, False),
    ("confirmed", params.delete_threshold - 0.01, 1.0, True),
    ("tentative", 0.5, params.max_P, False),
    ("tentative", 0.5, params.max_P + 0.01, True),
])
def test_student_lifecycle_delete_boundaries(
    workspace_modules, state, score, variance, expected
):
    covariance = np.eye(6)
    covariance[1, 1] = variance
    result = workspace_modules["track_management"].should_delete_track(
        {"score": score, "state": state, "P": np.asmatrix(covariance)}
    )
    assert bool(result) is expected


@pytest.mark.student_exercise
def test_association_checks_visibility_before_distance(workspace_modules, sensors, monkeypatch):
    _, camera = sensors
    association = workspace_modules["association"]
    camera.in_fov = lambda x: False
    def forbidden_distance(track, meas):
        pytest.fail("Invisible pairs must not evaluate projection/Mahalanobis distance")
    monkeypatch.setattr(association, "mahalanobis_distance", forbidden_distance)
    track = SimpleNamespace(x=_state([-1, 0, 0]), P=np.asmatrix(np.eye(6)))
    meas = Measurement(0, [400, 300], camera)
    cost = association.association_cost_matrix([track], [meas])
    assert cost.shape == (1, 1)
    assert np.isinf(cost[0, 0])


@pytest.mark.student_exercise
def test_association_empty_pass_still_manages_tracks(workspace_modules, sensors):
    lidar, _ = sensors
    calls = []
    track = SimpleNamespace(x=_state([10, 0, 0]))
    manager = SimpleNamespace(
        track_list=[track],
        manage_tracks=lambda tracks, measurements, sensor: calls.append(
            (tracks, measurements, sensor)
        ),
    )
    workspace_modules["association"].associate_and_update(manager, [], None, lidar)
    assert calls == [([track], [], lidar)]


@pytest.mark.student_exercise
@pytest.mark.parametrize("associated, state, score, expected_state, expected_score", [
    (True, "tentative", 1.0, "confirmed", 1.0),
    (True, "tentative", params.confirmed_threshold - 1 / params.window,
     "tentative", params.confirmed_threshold),
    (False, "confirmed", 1.0, "confirmed", 1.0 - 1 / params.window),
])
def test_student_score_saturation_and_confirmation_boundaries(
    workspace_modules, associated, state, score, expected_state, expected_score
):
    result = workspace_modules["track_management"].update_track_score(
        {"score": score, "state": state}, associated
    )
    assert result["score"] == pytest.approx(expected_score)
    assert result["state"] == expected_state


@pytest.mark.student_exercise
def test_association_preserves_unmatched_lists_when_all_pairs_rejected(workspace_modules):
    tracks, measurements = [object()], [object()]
    matrix = np.asmatrix([[np.inf]])
    track, meas, _, remaining_tracks, remaining_meas = workspace_modules[
        "association"
    ].pick_next_pair(matrix, tracks, measurements)
    assert np.isnan(track)
    assert np.isnan(meas)
    assert remaining_tracks == tracks
    assert remaining_meas == measurements


@pytest.mark.student_exercise
def test_association_does_not_recheck_visibility_after_pair_removal(
    workspace_modules, sensors, monkeypatch
):
    lidar, _ = sensors
    association = workspace_modules["association"]
    track = SimpleNamespace(x=_state([10, 0, 0]))
    meas = Measurement(0, [10, 0, 0, 1, 2, 4, 0], lidar)
    monkeypatch.setattr(association, "association_cost_matrix",
                        lambda tracks, measurements: np.asmatrix([[0.0]]))
    def forbidden_visibility(x):
        pytest.fail("Visibility must be checked before removing the selected pair")
    lidar.in_fov = forbidden_visibility
    updates, hits, finished = [], [], []
    manager = SimpleNamespace(
        track_list=[track],
        handle_updated_track=lambda selected, sensor: hits.append((selected, sensor)),
        manage_tracks=lambda tracks, measurements, sensor: finished.append(
            (tracks, measurements, sensor)
        ),
    )
    filter_obj = SimpleNamespace(update=lambda selected, observation: updates.append(
        (selected, observation)
    ))
    association.associate_and_update(manager, [meas], filter_obj, lidar)
    assert updates == [(track, meas)]
    assert hits == [(track, lidar)]
    assert finished == [([], [], lidar)]
