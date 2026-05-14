import numpy as np
from numpy.testing import assert_array_equal

from relu import Relu


def test_forward():
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    assert_array_equal(Relu().forward(x), np.array([[1.0, 0.0], [0.0, 3.0]]))


def test_backward():
    relu = Relu()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    assert_array_equal(relu.backward(relu.forward(x)), np.array([[1.0, 0.0], [0.0, 3.0]]))
