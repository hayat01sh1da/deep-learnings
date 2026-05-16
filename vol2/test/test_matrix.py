import numpy as np
import pytest
from numpy.testing import assert_array_equal

from matrix import Matrix


@pytest.fixture
def matrix():
    return Matrix(np.array([[1, 2, 3], [4, 5, 6]]))


def test_get_class_name(matrix):
    assert str(matrix.W.__class__) == "<class 'numpy.ndarray'>"


def test_get_shape(matrix):
    assert matrix.W.shape == (2, 3)


def test_get_dim(matrix):
    assert matrix.W.ndim == 2


def test_calc_sum(matrix):
    assert_array_equal(
        matrix.calc_sum(np.array([[0, 1, 2], [3, 4, 5]])),
        np.array([[1, 3, 5], [7, 9, 11]]),
    )


def test_calc_product(matrix):
    assert_array_equal(
        matrix.calc_product(np.array([[0, 1, 2], [3, 4, 5]])),
        np.array([[0, 2, 6], [12, 20, 30]]),
    )


def test_calc_scala_broadcast(matrix):
    assert_array_equal(
        matrix.calc_scala_broadcast(10),
        np.array([[10, 20, 30], [40, 50, 60]]),
    )


def test_calc_array_broadcast(matrix):
    assert_array_equal(
        matrix.calc_array_broadcast(np.array([10, 20, 30])),
        np.array([[10, 40, 90], [40, 100, 180]]),
    )


def test_calc_inner_product(matrix):
    assert_array_equal(
        matrix.calc_inner_product(np.array([[7, 8], [9, 10], [11, 12]])),
        np.array([[58, 64], [139, 154]]),
    )
