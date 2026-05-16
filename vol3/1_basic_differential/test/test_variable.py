import numpy as np
from numpy.testing import assert_array_equal

from variable import Variable


def test_variable_scalar():
    variable = Variable(np.array(1))
    assert_array_equal(variable.get_data(), np.array(1))
    assert variable.get_data().ndim == 0
    variable.set_data(np.array([10]))
    assert_array_equal(variable.get_data(), np.array(10))


def test_variable_1d():
    variable = Variable(np.array([1, 2, 3]))
    assert_array_equal(variable.get_data(), np.array([1, 2, 3]))
    assert variable.get_data().ndim == 1
    variable.set_data(np.array([1.0, 2.0, 3.0]))
    assert_array_equal(variable.get_data(), np.array([1.0, 2.0, 3.0]))


def test_variable_2d():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    variable = Variable(data)
    assert_array_equal(variable.get_data(), data)
    assert variable.get_data().ndim == 2
    new_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    variable.set_data(new_data)
    assert_array_equal(variable.get_data(), new_data)
