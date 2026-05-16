import numpy as np
import pytest

from time_attention import TimeAttention


@pytest.fixture
def setup():
    return TimeAttention(), np.random.randn(10, 4, 4), np.random.randn(10, 5, 4)


def test_forward(setup):
    time_attention, hs_enc, hs_dec = setup
    assert time_attention.forward(hs_enc, hs_dec).shape == (10, 5, 4)


def test_backward(setup):
    time_attention, hs_enc, hs_dec = setup
    dout = time_attention.forward(hs_enc, hs_dec)
    dhs_enc, dhs_dec = time_attention.backward(dout)
    assert dhs_enc.shape == (10, 4, 4)
    assert dhs_dec.shape == (10, 5, 4)
