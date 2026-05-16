import numpy as np
import pytest

from count_based_methods import CountBasedMethod
from simple_rnnlm import SimpleRNNLM


@pytest.fixture
def setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, *_ = cbm.preprocess(word_list)
    rnnlm = SimpleRNNLM(len(word_to_id), 100, 100)
    xs = np.array([[0, 4, 4, 1], [4, 0, 2, 1]])
    ts = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    return rnnlm, xs, ts


def test_predict(setup):
    rnnlm, xs, _ = setup
    assert rnnlm._predict(xs).shape == (2, 4, 7)


def test_forward(setup):
    rnnlm, xs, ts = setup
    assert 1.93 < round(rnnlm.forward(xs, ts), 2) < 1.96


def test_backward(setup):
    rnnlm, xs, ts = setup
    rnnlm.forward(xs, ts)
    assert rnnlm.backward() is None


def test_reset_state(setup):
    rnnlm, xs, ts = setup
    rnnlm.forward(xs, ts)
    rnnlm.backward()
    assert rnnlm.rnn_layer.h.shape == (2, 100)
    rnnlm.reset_state()
    assert rnnlm.rnn_layer.h is None
