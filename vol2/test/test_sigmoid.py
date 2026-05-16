import numpy as np
import pytest

from sigmoid import Sigmoid


@pytest.fixture
def setup():
    return Sigmoid(), np.random.randn(10, 4)


def test_forward(setup):
    sigmoid, x = setup
    assert sigmoid.forward(x).shape == (10, 4)


def test_backward(setup):
    sigmoid, x = setup
    sigmoid.forward(x)
    assert sigmoid.backward(np.random.randn(10, 4)).shape == (10, 4)
