import numpy as np
import pytest
from numpy.testing import assert_array_equal

from embedding_dot import EmbeddingDot


@pytest.fixture
def setup():
    W = np.arange(21).reshape(7, 3)
    embedding_dot = EmbeddingDot(W)
    return embedding_dot, np.array([0, 3, 1]), np.arange(9).reshape(3, 3)


def test_params(setup):
    embedding_dot, _, _ = setup
    params, = embedding_dot.params
    assert_array_equal(params, np.arange(21).reshape(7, 3))


def test_initial_grads(setup):
    embedding_dot, _, _ = setup
    grads, = embedding_dot.grads
    assert_array_equal(grads, np.zeros((7, 3)))


def test_forward(setup):
    embedding_dot, index, h = setup
    assert_array_equal(embedding_dot.forward(h, index), np.array([5, 122, 86]))


def test_h(setup):
    embedding_dot, index, h = setup
    embedding_dot.forward(h, index)
    h_cache, *_ = embedding_dot.cache
    assert_array_equal(h_cache, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))


def test_target_W(setup):
    embedding_dot, index, h = setup
    embedding_dot.forward(h, index)
    *_, target_W = embedding_dot.cache
    assert_array_equal(target_W, np.array([[0, 1, 2], [9, 10, 11], [3, 4, 5]]))


def test_backward(setup):
    embedding_dot, index, h = setup
    dout = embedding_dot.forward(h, index)
    dh = embedding_dot.backward(dout)
    assert_array_equal(
        dh,
        np.array([
            [0, 5, 10],
            [1098, 1220, 1342],
            [258, 344, 430],
        ]),
    )


def test_dtarget_W(setup):
    embedding_dot, index, h = setup
    dout = embedding_dot.forward(h, index)
    embedding_dot.backward(dout)
    assert_array_equal(
        embedding_dot.cache,
        np.array([
            [0, 5, 10],
            [366, 488, 610],
            [516, 602, 688],
        ]),
    )


def test_grads(setup):
    embedding_dot, index, h = setup
    dout = embedding_dot.forward(h, index)
    embedding_dot.backward(dout)
    grads, = embedding_dot.grads
    assert_array_equal(
        grads,
        np.array([
            [0, 5, 10],
            [516, 602, 688],
            [0, 0, 0],
            [366, 488, 610],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]),
    )
