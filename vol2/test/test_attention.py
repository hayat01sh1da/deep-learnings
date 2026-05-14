import numpy as np
import pytest

from attention import Attention


@pytest.fixture
def attention_setup():
    attention = Attention()
    hs = np.random.randn(10, 5, 4)
    h = np.random.randn(10, 4)
    return attention, hs, h


def test_forward(attention_setup):
    attention, hs, h = attention_setup
    assert attention.forward(hs, h).shape == (10, 4)


def test_backward(attention_setup):
    attention, hs, h = attention_setup
    dout = attention.forward(hs, h)
    dhs, dh = attention.backward(dout)
    assert dhs.shape == (10, 5, 4)
    assert dh.shape == (10, 5)
