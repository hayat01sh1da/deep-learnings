import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from softmax_with_loss import SoftMaxWithLoss


@pytest.fixture
def setup():
    softmax_with_loss = SoftMaxWithLoss()
    x = np.array([
        [-0.27291637, 3.0623984, 1.08772839, 1.21167545],
        [0.77815361, 1.20011612, -0.37179735, 1.93945452],
        [-1.02360881, -0.23723418, -1.42713268, -0.6484095],
        [-0.6631865, 0.01433258, -2.450729, -2.02298841],
    ])
    t = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ])
    return softmax_with_loss, x, t


def test_forward(setup):
    softmax_with_loss, x, t = setup
    assert softmax_with_loss.forward(x, t) == 0.5879459780961449


def test_backward(setup):
    softmax_with_loss, x, t = setup
    softmax_with_loss.forward(x, t)
    assert_almost_equal(
        softmax_with_loss.backward(),
        np.array([
            [0.00668465, -0.06224663, 0.0260615, 0.02950047],
            [0.04142029, 0.06316393, 0.01311583, -0.11770004],
            [0.04700427, -0.14680577, 0.03139707, 0.06840443],
            [0.0736796, -0.10492584, 0.01233183, 0.01891441],
        ]),
    )
