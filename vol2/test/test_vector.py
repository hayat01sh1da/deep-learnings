import numpy as np
import pytest

from vector import Vector


@pytest.fixture
def vector():
    return Vector(np.array([1, 2, 3]))


def test_get_class_name(vector):
    assert isinstance(vector.x, np.ndarray)


def test_get_shape(vector):
    assert vector.x.shape == (3,)


def test_get_dim(vector):
    assert vector.x.ndim == 1


def test_calc_inner_product(vector):
    assert vector.calc_inner_product(np.array([4, 5, 6])) == 32
