import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from time_softmax_with_loss import TimeSoftmaxWithLoss


@pytest.fixture
def setup():
    time_softmax_with_loss = TimeSoftmaxWithLoss()
    xs = np.array([
        [[3, 1, 3], [1, 3, 0], [2, 1, 1]],
        [[2, 4, 3], [0, 1, 0], [0, 4, 2]],
        [[2, 1, 0], [3, 2, 3], [1, 0, 2]],
    ])
    ts = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ])
    return time_softmax_with_loss, xs, ts


def test_forward(setup):
    layer, xs, ts = setup
    assert layer.forward(xs, ts) == 1.1399003832212435


def test_backward(setup):
    layer, xs, ts = setup
    layer.forward(xs, ts)
    assert_almost_equal(
        layer.backward(),
        np.array([
            [[-0.05907661, 0.0070421, 0.0520345], [0.01268836, -0.01735614, 0.00466779], [0.06401299, 0.02354906, -0.08756205]],
            [[-0.10110771, 0.07391566, 0.02719205], [0.02354906, -0.04709812, 0.02354906], [0.00176403, 0.09631259, -0.09807662]],
            [[-0.03719545, 0.02719205, 0.0100034], [0.04692431, -0.09384862, 0.04692431], [0.02719205, 0.0100034, -0.03719545]],
        ]),
    )
