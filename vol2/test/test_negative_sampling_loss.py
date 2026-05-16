import numpy as np
import pytest
from numpy.testing import assert_array_equal

from negative_sampling_loss import NegativeSamplingLoss


@pytest.fixture
def setup():
    W = np.arange(21).reshape(7, 3)
    corpus = np.array([0, 1, 2, 3, 4, 1, 5, 2, 6])
    nsl = NegativeSamplingLoss(W, corpus)
    h = np.arange(3)
    target = np.array([1, 3, 0])
    return nsl, h, target


def test_params(setup):
    nsl, _, _ = setup
    param, *_ = nsl.params
    assert_array_equal(param, np.arange(21).reshape(7, 3))


def test_initial_grads(setup):
    nsl, _, _ = setup
    grad, *_ = nsl.grads
    assert_array_equal(grad, np.zeros((7, 3)))


def test_forward(setup):
    nsl, h, target = setup
    assert 65 <= nsl.forward(h, target) < 75


def test_backward(setup):
    nsl, h, target = setup
    nsl.forward(h, target)
    assert_array_equal(
        np.round(nsl.backward(), decimals=1),
        np.array([
            [14.0, 15.7, 17.3],
            [12.0, 13.7, 15.3],
            [17.0, 18.7, 20.3],
        ]),
    )
