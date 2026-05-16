import numpy as np
import pytest
from numpy.testing import assert_array_equal

from embedding import Embedding


@pytest.fixture
def setup():
    W = np.arange(21).reshape(7, 3)
    return Embedding(W), np.array([0, 2, 0, 4])


def test_params(setup):
    embedding, _ = setup
    params, = embedding.params
    assert_array_equal(params, np.arange(21).reshape(7, 3))


def test_grads(setup):
    embedding, _ = setup
    grads, = embedding.grads
    assert_array_equal(grads, np.zeros((7, 3)))


def test_forward(setup):
    embedding, index = setup
    assert_array_equal(
        embedding.forward(index),
        np.array([
            [0, 1, 2],
            [6, 7, 8],
            [0, 1, 2],
            [12, 13, 14],
        ]),
    )


def test_backward(setup):
    embedding, index = setup
    dout = embedding.forward(index)
    embedding.backward(dout)
    grads, = embedding.grads
    assert_array_equal(
        grads,
        np.array([
            [0, 2, 4],
            [0, 0, 0],
            [6, 7, 8],
            [0, 0, 0],
            [12, 13, 14],
            [0, 0, 0],
            [0, 0, 0],
        ]),
    )
