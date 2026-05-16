from neural_network import NeuralNetwork
import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def nnw():
    return NeuralNetwork()


def test_sigmoid(nnw):
    x = np.array([-1.0, 0.0, 1.0])
    assert_almost_equal(nnw._sigmoid(x), 1.0 / (1.0 + np.exp(-x)))


def test_softmax(nnw):
    a = np.array([0.3, 2.9, 4.0])
    assert_almost_equal(nnw._softmax(a), np.array(
        [0.01821127, 0.24519181, 0.73659691]))


def test_step_relu_matrix(nnw):
    x = np.array([-1.0, 0.0, 1.0])
    assert_array_equal(nnw.step_func(x), np.array([0, 0, 1]))
    assert_array_equal(nnw.relu(x), np.array([0.0, 0.0, 1.0]))

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    assert_array_equal(nnw.matrix_product(
        a, b), np.array([[19, 22], [43, 50]]))
