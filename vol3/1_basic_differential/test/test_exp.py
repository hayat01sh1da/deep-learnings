import numpy as np
import pytest

from exp import Exp
from square import Square
from variable import Variable


@pytest.fixture
def exp():
    return Exp()


@pytest.fixture
def input_value():
    return np.array(0.5)


def test_call(exp, input_value):
    square_1 = Square()
    square_2 = Square()
    x = Variable(input_value)
    a = square_1(x)
    b = exp(a)
    y = square_2(b)
    assert a.data == 0.25
    assert b.data == 1.2840254166877414
    assert y.data == 1.648721270700128


def test_forward(exp, input_value):
    assert exp.forward(input_value) == 1.6487212707001282
