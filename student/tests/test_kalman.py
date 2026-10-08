import pytest

import numpy as np


pytestmark = pytest.mark.student_exercise

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
