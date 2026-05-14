import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from neural_network_learning import NeuralNetworkLearning


@pytest.fixture
def nwl():
    return NeuralNetworkLearning()


@pytest.mark.parametrize(
    ('y', 'expected'),
    [
        ([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], 0.09750000000000003),
        ([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], 0.5975),
    ],
)
def test_mean_squared_error(nwl, y, expected):
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert nwl.mean_squared_error(np.array(y), np.array(t)) == expected


@pytest.mark.parametrize(
    ('y', 'expected'),
    [
        ([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], 0.510825457099338),
        ([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], 2.302584092994546),
    ],
)
def test_cross_entropy_error(nwl, y, expected):
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert nwl.cross_entropy_error(np.array(y), np.array(t)) == expected


def _function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


@pytest.mark.parametrize(
    ('x', 'expected'),
    [(5, 0.1999999999990898), (10, 0.2999999999986347)],
)
def test_numerical_diff_function_1(nwl, x, expected):
    assert nwl.numerical_diff(_function_1, x) == expected


def _function_2(x):
    return np.sum(x ** 2)


@pytest.mark.parametrize(
    ('x', 'expected'),
    [(5, 9.999999999976694), (10, 19.99999999995339)],
)
def test_numerical_diff_function_2(nwl, x, expected):
    assert nwl.numerical_diff(_function_2, x) == expected


def test_numerical_diff_function_3(nwl):
    assert nwl.numerical_diff(lambda x0: x0 * x0 + 4.0 ** 2.0, 3.0) == 6.00000000000378


def test_numerical_diff_function_4(nwl):
    assert nwl.numerical_diff(lambda x1: 3.0 ** 2.0 + x1 * x1, 4.0) == 7.999999999999119


@pytest.mark.parametrize(
    ('point', 'expected'),
    [
        ([3.0, 4.0], [6.0, 8.0]),
        ([0.0, 2.0], [0.0, 4.0]),
        ([3.0, 0.0], [6.0, 0.0]),
    ],
)
def test_numerial_gradient(nwl, point, expected):
    assert_almost_equal(nwl.numerial_gradient(_function_2, np.array(point)), np.array(expected))


def test_gradient_descent(nwl):
    result = nwl.gradient_descent(_function_2, np.array([3.0, 4.0]), learning_rate=0.1)
    assert_almost_equal(result, np.array([6.11110793e-10, 8.14814391e-10]))
