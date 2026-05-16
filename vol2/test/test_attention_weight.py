import numpy as np
import pytest

from attention_weight import AttentionWeight


@pytest.fixture
def weight_setup():
    return AttentionWeight(), np.random.randn(10, 5, 4), np.random.randn(10, 4)


def test_forward(weight_setup):
    weight, hs, h = weight_setup
    assert weight.forward(hs, h).shape == (10, 5)


def test_backward(weight_setup):
    weight, hs, h = weight_setup
    da = weight.forward(hs, h)
    dhs, da = weight.backward(da)
    assert dhs.shape == (10, 5, 4)
    assert da.shape == (10, 5)
