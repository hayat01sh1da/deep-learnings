import numpy as np
import pytest

from rnn import RNN


@pytest.fixture
def setup():
    Wx = np.random.randn(3, 3)
    Wh = np.random.randn(7, 3)
    b = np.random.randn(3)
    return RNN(Wx, Wh, b), np.random.randn(7, 3), np.random.randn(7, 7)


def test_forward(setup):
    rnn, x, h_prev = setup
    assert rnn.forward(x, h_prev).shape == (7, 3)


def test_backward(setup):
    rnn, x, h_prev = setup
    h_next = rnn.forward(x, h_prev)
    dx, dh_prev = rnn.backward(h_next)
    assert dx.shape == (7, 3)
    assert dh_prev.shape == (7, 7)
