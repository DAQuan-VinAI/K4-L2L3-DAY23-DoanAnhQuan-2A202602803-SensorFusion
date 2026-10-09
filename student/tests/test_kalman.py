import numpy as np


def test_build_F_shape(workspace_modules):
    k = workspace_modules["kalman"]
    F = k.build_F(dt=0.1)
    assert F.shape == (6, 6)
    assert F[0, 3] == 0.1


def test_ekf_predict(workspace_modules):
    k = workspace_modules["kalman"]
    x = np.asmatrix(np.zeros((6, 1)))
    P = np.asmatrix(np.eye(6))
    x2, P2 = k.ekf_predict(x, P)
    assert x2.shape == (6, 1)
    assert P2.shape == (6, 6)


def test_predict_numeric_state_and_covariance(workspace_modules):
    k = workspace_modules["kalman"]
    state = np.asmatrix([[1], [2], [3], [4], [5], [6]], dtype=float)
    covariance = np.asmatrix(np.diag([1, 2, 3, 4, 5, 6]))
    transition = k.build_F(dt=0.2)
    noise = k.build_Q(dt=0.2, q=3)
    predicted, predicted_covariance = k.ekf_predict(
        state, covariance, F=transition, Q=noise
    )
    np.testing.assert_allclose(predicted, [[1.8], [3], [4.2], [4], [5], [6]])
    expected = np.diag([1.76, 2.8, 3.84, 4.6, 5.6, 6.6])
    expected[0, 3] = expected[3, 0] = 0.8
    expected[1, 4] = expected[4, 1] = 1.0
    expected[2, 5] = expected[5, 2] = 1.2
    np.testing.assert_allclose(predicted_covariance, expected)


def test_update_numeric_known_scalar_case(workspace_modules):
    from types import SimpleNamespace

    k = workspace_modules["kalman"]
    sensor = SimpleNamespace(
        get_H=lambda _: np.asmatrix([[1, 0, 0, 0, 0, 0]]),
        get_hx=lambda state: state[:1],
    )
    meas = SimpleNamespace(sensor=sensor, z=np.asmatrix([[12.0]]),
                           R=np.asmatrix([[1.0]]))
    state = np.asmatrix([[10], [2], [3], [0], [0], [0]], dtype=float)
    updated, covariance = k.ekf_update(state, np.asmatrix(np.eye(6)), meas)
    np.testing.assert_allclose(updated, [[11], [2], [3], [0], [0], [0]])
    np.testing.assert_allclose(covariance, np.diag([0.5, 1, 1, 1, 1, 1]))
