import numpy as np
import pytest

from diff import f, numerical_diff
from variable import Variable


@pytest.fixture
def x():
    return Variable(np.array(0.5))


def test_numerical_diff_1(x):
    assert numerical_diff(f, x) == 3.2974426293330694


def test_f(x):
    assert f(x).data == 1.648721270700128
