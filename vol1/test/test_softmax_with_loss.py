import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from softmax_with_loss import SoftmaxWithLoss


@pytest.fixture
def swl():
    return SoftmaxWithLoss()


def test_forward(swl):
    assert swl.forward(np.array([0.3, 0.6, 0.9]), np.array([0, 0, 1])) == 0.8283899409431649


def test_backward(swl):
    swl.forward(np.array([0.3, 0.6, 0.9]), np.array([0, 0, 1]))
    assert_almost_equal(swl.backward(), np.array([0.07989816, 0.10785123, -0.18774939]))
