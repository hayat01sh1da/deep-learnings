import numpy as np
import pytest

from repeat import Repeat


@pytest.fixture
def repeat():
    return Repeat(8, 7)


def test_forward(repeat):
    x = np.random.randn(1, repeat.D)
    assert repeat.forward(x).shape == (7, 8)


def test_backward(repeat):
    dy = np.random.randn(repeat.N, repeat.D)
    assert repeat.backward(dy).shape == (1, 8)
