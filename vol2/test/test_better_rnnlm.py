import numpy as np
import pytest

from better_rnnlm import BetterRNNLM
from count_based_methods import CountBasedMethod


@pytest.fixture
def rnnlm_setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, *_ = cbm.preprocess(word_list)
    rnnlm = BetterRNNLM(len(word_to_id), 100, 100)
    xs = np.array([[0, 4, 4, 1], [4, 0, 2, 1]])
    ts = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    return rnnlm, xs, ts


def test_predict(rnnlm_setup):
    rnnlm, xs, _ = rnnlm_setup
    assert rnnlm._predict(xs).shape == (2, 4, 7)


def test_forward(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    assert round(rnnlm.forward(xs, ts), 2) == 1.95


def test_backward(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    rnnlm.forward(xs, ts)
    assert rnnlm.backward() is None


def test_reset_state(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    rnnlm.forward(xs, ts)
    rnnlm.backward()
    assert rnnlm.lstm_layers[0].h.shape == (2, 100)
    rnnlm.reset_state()
    assert rnnlm.lstm_layers[0].h is None
