import numpy as np
import pytest

from two_layer_net import TwoLayerNet


@pytest.fixture
def setup():
    return TwoLayerNet(2, 4, 3), np.random.randn(4, 2), np.random.randn(4, 3)


def test_predict(setup):
    net, x, _ = setup
    assert net._predict(x).shape == (4, 3)


def test_forward(setup):
    net, x, t = setup
    assert int(net.forward(x, t)) == 1


def test_backward(setup):
    net, x, t = setup
    net.forward(x, t)
    assert net.backward().shape == (4, 2)
