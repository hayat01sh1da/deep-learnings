import numpy as np
import pytest

from function import Function
from variable import Variable


def test_call_raises_not_implemented():
    data = np.array([
        [0, 4, 0],
        [0, 2, 5],
        [9, 9, 1],
    ])
    with pytest.raises(NotImplementedError):
        Function()(Variable(data))
