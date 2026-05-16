import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from affine import Affine


@pytest.fixture
def affine():
    W = np.array([
        [-0.22472106, -0.42868683, 0.21713442],
        [-0.13635294, 0.45327181, -1.31839392],
    ])
    b = np.array([1.55270156, 1.44441689, -1.69451485])
    return Affine(W, b)


def test_forward(affine):
    x = np.array([1.52949391, -0.81788271])
    assert_almost_equal(
        affine.forward(x), [
            1.32051278, 0.41801982, -0.28411748])


def test_backward(affine):
    x = np.array([1.52949391, -0.81788271])
    affine.forward(x)
    assert_almost_equal(
        affine.backward(1),
        np.array([
            [-0.2247211, -0.1363529],
            [-0.4286868, 0.4532718],
            [0.2171344, -1.3183939],
        ]),
    )
