import numpy as np
import pytest

from sum import Sum


@pytest.fixture
def sum_layer():
    return Sum(8, 7)


def test_forward(sum_layer):
    x = np.random.randn(sum_layer.N, sum_layer.D)
    assert sum_layer.forward(x).shape == (1, 8)


def test_backward(sum_layer):
    dy = np.random.randn(1, sum_layer.D)
    assert sum_layer.backward(dy).shape == (7, 8)
