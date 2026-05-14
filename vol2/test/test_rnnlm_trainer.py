import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ptb import load_data
from rnnlm_trainer import RNNLMTrainer
from sgd import SGD
from simple_rnnlm import SimpleRNNLM


BATCH_SIZE = 10
TIME_SIZE = 5
MAX_EPOCH = 100


@pytest.fixture
def setup():
    sys_corpus, _, _ = load_data('train')
    corpus = sys_corpus[:1000]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]
    ts = corpus[1:]
    model = SimpleRNNLM(vocab_size, 100, 100)
    trainer = RNNLMTrainer(model, SGD(0.1))
    return trainer, xs, ts


def test_get_batch(setup):
    trainer, xs, ts = setup
    batch_x, batch_t = trainer._get_batch(xs, ts, BATCH_SIZE, TIME_SIZE)
    assert_array_equal(batch_x, np.array([
        [0, 1, 2, 3, 4],
        [42, 76, 77, 64, 78],
        [26, 26, 98, 56, 40],
        [24, 32, 26, 175, 98],
        [208, 209, 80, 197, 32],
        [26, 79, 26, 80, 32],
        [274, 275, 276, 42, 61],
        [88, 303, 26, 304, 26],
        [42, 35, 72, 350, 64],
        [339, 359, 181, 328, 386],
    ]))
    assert_array_equal(batch_t, np.array([
        [1, 2, 3, 4, 5],
        [76, 77, 64, 78, 79],
        [26, 98, 56, 40, 128],
        [32, 26, 175, 98, 61],
        [209, 80, 197, 32, 82],
        [79, 26, 80, 32, 241],
        [275, 276, 42, 61, 24],
        [303, 26, 304, 26, 32],
        [35, 72, 350, 64, 27],
        [359, 181, 328, 386, 387],
    ]))


def test_fit(setup):
    trainer, xs, ts = setup
    assert len(trainer.fit(xs, ts, MAX_EPOCH, BATCH_SIZE, TIME_SIZE)) == 100


def test_save_plot_image(setup):
    trainer, xs, ts = setup
    trainer.fit(xs, ts, MAX_EPOCH, BATCH_SIZE, TIME_SIZE)
    file_path = '../img/rnnlm_trainer.png'
    trainer.save_plot_image(file_path)
    assert os.path.exists(file_path)
