import numpy as np
from numpy.testing import assert_almost_equal

from sigmoid import Sigmoid


def test_forward():
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    assert_almost_equal(
        Sigmoid().forward(x),
        [[0.73105858, 0.37754067], [0.11920292, 0.95257413]],
    )


def test_backward():
    sigmoid = Sigmoid()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    sigmoid.forward(x)
    dout = 1
    assert_almost_equal(
        sigmoid.backward(sigmoid.backward(dout)),
        np.array([[0.0386563, 0.0552267], [0.0110237, 0.0020409]]),
    )
