import numpy as np
import pytest

from time_affine import TimeAffine


@pytest.fixture
def setup():
    W = np.random.randn(3, 3)
    b = np.random.randn(3)
    return TimeAffine(W, b), np.random.randn(3, 3, 3)


def test_forward(setup):
    time_affine, x = setup
    assert time_affine.forward(x).shape == (3, 3, 3)


def test_backward(setup):
    time_affine, x = setup
    dout = time_affine.forward(x)
    time_affine.backward(dout)
    assert dout.shape == (3, 3, 3)
