import numpy as np
import pytest

from seq2seq import Seq2Seq


@pytest.fixture
def setup():
    seq2seq = Seq2Seq(13, 100, 100)
    xs = np.random.randint(0, 13, (13, 100))
    ts = np.random.randint(0, 13, (13, 100))
    return seq2seq, xs, ts


def test_forward(setup):
    seq2seq, xs, ts = setup
    assert 2.55 < round(seq2seq.forward(xs, ts), 2) < 2.58


def test_backward(setup):
    seq2seq, xs, ts = setup
    seq2seq.forward(xs, ts)
    assert seq2seq.backward() is None


def test_generate(setup):
    seq2seq, _, _ = setup
    xs = np.random.randint(0, 13, (1, 100))
    assert len(seq2seq.generate(xs, 0, 10)) == 10
