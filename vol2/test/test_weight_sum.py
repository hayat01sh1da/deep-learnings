import numpy as np
import pytest

from weight_sum import WeightSum


@pytest.fixture
def setup():
    return WeightSum(), np.random.randn(10, 5, 4), np.random.randn(10, 5)


def test_forward(setup):
    weight_sum, hs, a = setup
    assert weight_sum.forward(hs, a).shape == (10, 4)


def test_backward(setup):
    weight_sum, hs, a = setup
    dc = weight_sum.forward(hs, a)
    dhs, da = weight_sum.backward(dc)
    assert dhs.shape == (10, 5, 4)
    assert da.shape == (10, 5)
