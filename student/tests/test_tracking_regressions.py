"""Numeric self-checks for camera, association, and lidar-driven lifecycle."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import Track
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Measurement
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_support import get_tracking_params


def _state(x=10, y=0, z=0):
    return np.asmatrix([[x], [y], [z], [0], [0], [0]], dtype=float)


def _camera(workspace_modules):
    # A calibrated camera with a nontrivial extrinsic rotation and translation.
    transform = np.eye(4)
    theta = 0.17
    transform[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]]
    transform[:3, 3] = [0.5, -0.1, 0.2]
    calibration = SimpleNamespace(
        extrinsic=SimpleNamespace(transform=transform.reshape(-1).tolist()),
        intrinsic=[800, 810, 960, 640], width=1920, height=1280,
    )
    return Sensor("camera", calibration, workspace_modules["camera_fusion"])


def _identity_camera(workspace_modules):
    sensor = _camera(workspace_modules)
    sensor.veh_to_sens = np.asmatrix(np.eye(4))
    return sensor


def _lidar(workspace_modules):
    return Sensor("lidar", None, workspace_modules["camera_fusion"])


@pytest.mark.parametrize("xyz", [(0, 0, 0), (-1, 0, 0), (1e-7, 0, 0),
                                     (10, np.nan, 0), (np.inf, 0, 0)])
def test_camera_invalid_domain(workspace_modules, xyz):
    sensor = _identity_camera(workspace_modules)
    state = _state(*xyz)
    assert not sensor.in_fov(state)
    with pytest.raises(ValueError, match="Camera|camera"):
        sensor.get_hx(state)
    with pytest.raises(ValueError, match="Camera|camera"):
        sensor.get_H(state)


def test_camera_front_projection_and_noise(workspace_modules):
    sensor = _identity_camera(workspace_modules)
    state = _state(10, 1, 2)
    assert sensor.in_fov(state)
    np.testing.assert_allclose(sensor.get_hx(state), [[880], [478]])
    payload = workspace_modules["camera_fusion"].build_camera_measurement([880, 478], sensor)
    params = get_tracking_params()
    np.testing.assert_allclose(payload["z"], [[880], [478]])
    np.testing.assert_allclose(payload["R"], np.diag([params.sigma_cam_i**2,
                                                     params.sigma_cam_j**2]))


def test_camera_jacobian_matches_finite_differences(workspace_modules):
    sensor = _camera(workspace_modules)
    state = _state(12, 0.8, 1.4)
    jacobian = sensor.get_H(state)
    numerical = np.zeros((2, 6))
    for index in range(6):
        perturbation = np.zeros((6, 1))
        perturbation[index] = 1e-5
        numerical[:, index] = np.asarray(
            (sensor.get_hx(state + perturbation) -
             sensor.get_hx(state - perturbation)) / 2e-5
        ).reshape(2)
    np.testing.assert_allclose(jacobian, numerical, rtol=1e-6, atol=1e-7)


def test_out_of_fov_gating_precedes_projection(workspace_modules):
    sensor = _identity_camera(workspace_modules)
    track = SimpleNamespace(x=_state(-1), P=np.asmatrix(np.eye(6)))
    measurement = Measurement(0, [960, 640], sensor,
                              R=np.asmatrix(np.eye(2)))
    sensor.get_H = Mock(side_effect=AssertionError("Invalid projection evaluated"))
    sensor.get_hx = Mock(side_effect=AssertionError("Invalid projection evaluated"))
    costs = workspace_modules["association"].association_cost_matrix([track], [measurement])
    assert costs.shape == (1, 1)
    assert np.isinf(costs[0, 0])
    sensor.get_H.assert_not_called()
    sensor.get_hx.assert_not_called()


def test_out_of_fov_pair_stays_unassigned(workspace_modules):
    association = workspace_modules["association"]
    sensor = _identity_camera(workspace_modules)
    track = SimpleNamespace(x=_state(-1), P=np.asmatrix(np.eye(6)))
    measurement = Measurement(0, [960, 640], sensor, R=np.asmatrix(np.eye(2)))
    manager = SimpleNamespace(track_list=[track], handle_updated_track=Mock(),
                              manage_tracks=Mock())
    filter_obj = SimpleNamespace(update=Mock())
    association.associate_and_update(manager, [measurement], filter_obj, sensor)
    filter_obj.update.assert_not_called()
    manager.handle_updated_track.assert_not_called()
    manager.manage_tracks.assert_called_once_with([track], [measurement],
                                                  [measurement], sensor)


def test_empty_pass_runs_management(workspace_modules):
    track = SimpleNamespace(x=_state())
    sensor = _lidar(workspace_modules)
    manager = SimpleNamespace(track_list=[track], manage_tracks=Mock())
    workspace_modules["association"].associate_and_update(manager, [], Mock(), sensor)
    manager.manage_tracks.assert_called_once_with([track], [], [], sensor)


def test_mahalanobis_and_chi2_gate_numeric(workspace_modules):
    association = workspace_modules["association"]
    sensor = _lidar(workspace_modules)
    measurement = SimpleNamespace(sensor=sensor, z=np.asmatrix([[12], [0], [0]]),
                                  R=np.asmatrix(np.eye(3)))
    track = SimpleNamespace(x=_state(), P=np.asmatrix(np.eye(6)))
    assert association.mahalanobis_distance(track, measurement) == pytest.approx(2)
    assert association.chi2_gate(2, sensor)
    assert not association.chi2_gate(100, sensor)


def test_greedy_assignment_keeps_remaining_objects(workspace_modules):
    tracks, measurements = [object(), object()], [object(), object()]
    result = workspace_modules["association"].pick_next_pair(
        np.asmatrix([[4, 1], [2, np.inf]]), tracks, measurements
    )
    assert result[0] is tracks[0]
    assert result[1] is measurements[1]
    np.testing.assert_array_equal(result[2], [[2]])
    assert result[3] == [tracks[1]]
    assert result[4] == [measurements[0]]


def test_lidar_score_caps_and_confirmation_survives_one_miss(workspace_modules):
    management = workspace_modules["track_management"]
    track = {"score": 1.0, "state": "confirmed"}
    management.update_track_score(track, True)
    assert track == {"score": 1.0, "state": "confirmed"}
    management.update_track_score(track, False)
    assert track["score"] == pytest.approx(5 / 6)
    assert track["state"] == "confirmed"
    assert not management.should_delete_track({**track, "P": np.eye(6)})


@pytest.mark.parametrize(
    "state,score,variance,deleted",
    [("confirmed", 0.6, 9, False), ("confirmed", 0.59, 1, True),
     ("tentative", 0, 1, True), ("initialized", 0.01, 1, False),
     ("confirmed", 1, 9.01, True), ("tentative", 0.5, 9.01, True)],
)
def test_deletion_thresholds(workspace_modules, state, score, variance, deleted):
    covariance = np.eye(6)
    covariance[1, 1] = variance
    assert workspace_modules["track_management"].should_delete_track(
        {"state": state, "score": score, "P": covariance}
    ) is deleted


@pytest.mark.parametrize("camera_unmatched", [False, True])
def test_lidar_hits_confirm_despite_camera_empty_or_unmatched(workspace_modules,
                                                             camera_unmatched):
    lidar = _lidar(workspace_modules)
    camera = _identity_camera(workspace_modules)
    manager = TrackManager(workspace_modules["track_management"])
    filter_obj = Filter(workspace_modules["kalman"])
    association = workspace_modules["association"]
    for frame in range(8):
        lidar_meas = Measurement(frame, [10, 0, 0, 1.6, 2, 4, -0.3], lidar)
        association.associate_and_update(manager, [lidar_meas], filter_obj, lidar)
        before = [(track.score, track.state, track.id) for track in manager.track_list]
        camera_meas = ([Measurement(frame, [10000, 10000], camera,
                                   R=np.asmatrix(np.eye(2) * 25))]
                       if camera_unmatched else [])
        association.associate_and_update(manager, camera_meas, filter_obj, camera)
        assert [(track.score, track.state, track.id) for track in manager.track_list] == before
    assert len(manager.track_list) == 1
    assert manager.track_list[0].state == "confirmed"
    assert manager.track_list[0].score == pytest.approx(1)


def test_empty_lidar_frame_scores_then_deletes_exhausted_track(workspace_modules):
    lidar = _lidar(workspace_modules)
    manager = TrackManager(workspace_modules["track_management"])
    association = workspace_modules["association"]
    association.associate_and_update(
        manager, [Measurement(0, [10, 0, 0, 1.6, 2, 4, 0], lidar)],
        Filter(workspace_modules["kalman"]), lidar,
    )
    assert len(manager.track_list) == 1
    association.associate_and_update(manager, [], Mock(), lidar)
    assert manager.track_list == []


def test_camera_pass_never_deletes_or_spawns(workspace_modules):
    lidar = _lidar(workspace_modules)
    camera = _identity_camera(workspace_modules)
    manager = TrackManager(workspace_modules["track_management"])
    track = Track(Measurement(0, [10, 0, 0, 1.6, 2, 4, 0], lidar), 0,
                  workspace_modules["track_management"])
    track.score = 0
    track.P = np.asmatrix(np.eye(6) * 100)
    manager.track_list = [track]
    manager.manage_tracks([track], [], [], camera)
    assert manager.track_list == [track]
    manager.track_list = []
    meas = Measurement(0, [960, 640], camera, R=np.asmatrix(np.eye(2)))
    manager.manage_tracks([], [meas], [meas], camera)
    assert manager.track_list == []


def test_signed_yaw_and_frame_zero_timestamp(workspace_modules):
    lidar = _lidar(workspace_modules)
    measurement = Measurement(0, [10, 0, 0, 1.6, 2, 4, -0.3], lidar)
    track = Track(measurement, 0, workspace_modules["track_management"])
    assert measurement.t == pytest.approx(0)
    assert track.yaw == pytest.approx(-0.3)
    measurement.yaw = -0.7
    track.update_attributes(measurement)
    assert track.yaw == pytest.approx(-0.7)


def test_camera_hit_refines_state_without_changing_score(workspace_modules):
    lidar = _lidar(workspace_modules)
    camera = _identity_camera(workspace_modules)
    manager = TrackManager(workspace_modules["track_management"])
    track = Track(Measurement(0, [10, 0, 0, 1.6, 2, 4, 0], lidar), 0,
                  workspace_modules["track_management"])
    manager.track_list = [track]
    before_score, before_state = track.score, track.state
    measurement = Measurement(1, [958, 640], camera,
                              R=np.asmatrix(np.eye(2) * 25))
    workspace_modules["association"].associate_and_update(
        manager, [measurement], Filter(workspace_modules["kalman"]), camera
    )
    assert track.x[1, 0] > 0
    assert track.score == before_score
    assert track.state == before_state
    assert manager.track_list == [track]


def test_association_uses_workspace_kalman_not_plain_import(workspace_modules,
                                                          monkeypatch):
    import sys

    from fusion_lab.workspace_loader import load_workspace_module

    sensor = _lidar(workspace_modules)
    measurement = SimpleNamespace(sensor=sensor, z=np.asmatrix([[12], [0], [0]]),
                                  R=np.asmatrix(np.eye(3)))
    track = SimpleNamespace(x=_state(), P=np.asmatrix(np.eye(6)))
    stale_kalman = Mock(side_effect=AssertionError("Used stale plain kalman"))
    stale_kalman.innovation.side_effect = AssertionError("Used stale plain kalman")
    stale_kalman.innovation_covariance.side_effect = AssertionError(
        "Used stale plain kalman")
    # A stale plain ``kalman`` is importable before association loads, so a
    # plain import (module level or inside a function) would pick it up.
    monkeypatch.setitem(sys.modules, "kalman", stale_kalman)
    loaded = workspace_modules["association"].__name__
    package = loaded.rpartition(".")[0]
    monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.delattr(sys.modules[package], "association", raising=False)
    association = load_workspace_module("association")
    assert association.mahalanobis_distance(track, measurement) == pytest.approx(2)
    stale_kalman.innovation.assert_not_called()
    stale_kalman.innovation_covariance.assert_not_called()
