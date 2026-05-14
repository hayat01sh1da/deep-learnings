import numpy as np
import pytest
from numpy.testing import assert_array_equal

from square import Square
from variable import Variable


@pytest.fixture
def square():
    return Square()


def test_call(square):
    data = np.array([
        [0, 4, 0],
        [0, 2, 5],
        [9, 9, 1],
    ])
    output = square(Variable(data))
    assert_array_equal(
        output.data,
        np.array([
            [0, 16, 0],
            [0, 4, 25],
            [81, 81, 1],
        ]),
    )


def test_forward(square):
    x = np.array([
        [0, 4, 0],
        [0, 2, 5],
        [9, 9, 1],
    ])
    assert_array_equal(
        square.forward(x),
        np.array([
            [0, 16, 0],
            [0, 4, 25],
            [81, 81, 1],
        ]),
    )
