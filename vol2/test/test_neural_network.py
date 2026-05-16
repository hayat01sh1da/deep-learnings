import numpy as np
import pytest

from neural_network import NeuralNetwork


@pytest.fixture
def networks():
    x = np.random.randn(10, 2)
    W1 = np.random.randn(2, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4, 3)
    b2 = np.random.randn(3)
    return NeuralNetwork(x, W1, b1), NeuralNetwork(x, W2, b2)


def test_get_hidden_layer_dim(networks):
    network_1, _ = networks
    assert network_1.get_hidden_layer().shape == (10, 4)


def test_get_output_layer(networks):
    network_1, network_2 = networks
    h = network_1.get_hidden_layer()
    assert network_2.get_output_layer(h).shape == (10, 3)
