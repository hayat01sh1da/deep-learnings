import numpy as np
import pytest

from time_dropout import TimeDropout


@pytest.fixture
def setup():
    xs = np.array([
        [-2.02263879, 1.79293276, -0.64214657],
        [-0.74505721, -0.81903631, -0.27078458],
        [-0.16396182, -0.19952967, -0.88905705],
        [0.70678772, -0.71987105, 2.28794441],
        [-0.79917515, -0.53706248, 0.15224767],
        [-0.73616716, 0.05082873, -0.54139266],
        [-1.12802119, 1.3867995, 0.42552788],
    ])
    return TimeDropout(), xs


def test_forward(setup):
    time_dropout, xs = setup
    assert time_dropout.forward(xs).shape == (7, 3)


def test_backward(setup):
    time_dropout, xs = setup
    dout = time_dropout.forward(xs)
    assert time_dropout.backward(dout).shape == (7, 3)
