import numpy as np
import pytest

from count_based_methods import CountBasedMethod
from rnnlm import RNNLM


@pytest.fixture
def rnnlm_setup():
    # Seed so weight initialisation (np.random.randn) is deterministic.
    np.random.seed(0)
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, *_ = cbm.preprocess(word_list)
    rnnlm = RNNLM(len(word_to_id), 100, 100)
    xs = np.array([[0, 4, 4, 1], [4, 0, 2, 1]])
    ts = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    return rnnlm, xs, ts


def test_predict(rnnlm_setup):
    rnnlm, xs, _ = rnnlm_setup
    assert rnnlm._predict(xs).shape == (2, 4, 7)


def test_forward(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    assert round(rnnlm.forward(xs, ts), 2) == 1.94


def test_backward(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    rnnlm.forward(xs, ts)
    assert rnnlm.backward() is None


def test_reset_state(rnnlm_setup):
    rnnlm, xs, ts = rnnlm_setup
    rnnlm.forward(xs, ts)
    rnnlm.backward()
    assert rnnlm.lstm_layer.h.shape == (2, 100)
    rnnlm.reset_state()
    assert rnnlm.lstm_layer.h is None


def test_save_params(rnnlm_setup, tmp_path):
    rnnlm, xs, ts = rnnlm_setup
    rnnlm.forward(xs, ts)
    rnnlm.backward()
    file_path = tmp_path / 'rnnlm.pkl'
    rnnlm.save_params(str(file_path))
    assert file_path.exists()


def test_load_params(rnnlm_setup, tmp_path):
    rnnlm, xs, ts = rnnlm_setup
    file_path = tmp_path / 'rnnlm.pkl'
    rnnlm.save_params(str(file_path))
    rnnlm.load_params(str(file_path))
    a, b, c, d, e, f = rnnlm.params
    assert a.shape == (7, 100)
    assert b.shape == (100, 400)
    assert c.shape == (100, 400)
    assert d.shape == (400,)
    assert e.shape == (100, 7)
    assert f.shape == (7,)
