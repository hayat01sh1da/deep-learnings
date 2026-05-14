import numpy as np
import pytest
from numpy.testing import assert_array_equal

from time_rnn import TimeRNN


@pytest.fixture
def setup():
    Wx = np.random.randn(3, 3)
    Wh = np.random.randn(3, 3)
    b = np.random.randn(3)
    return TimeRNN(Wx, Wh, b), np.random.randn(3, 3, 3)


def test_state(setup):
    time_rnn, _ = setup
    h = np.random.randn(7, 7)
    time_rnn.set_state(h)
    assert_array_equal(time_rnn.h, h)
    time_rnn.reset_state()
    assert time_rnn.h is None


def test_forward(setup):
    time_rnn, xs = setup
    assert time_rnn.forward(xs).shape == (3, 3, 3)


def test_backward(setup):
    time_rnn, xs = setup
    hs = time_rnn.forward(xs)
    assert time_rnn.backward(hs).shape == (3, 3, 3)
