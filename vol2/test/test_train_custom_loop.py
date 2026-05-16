import os

import numpy as np
import pytest

from spiral import load_data
from train_custom_loop import TrainCustomLoop


MAX_EPOCH = 300
BATCH_SIZE = 30


@pytest.fixture
def setup():
    # Seed so TwoLayerNet weight initialisation is deterministic.
    np.random.seed(0)
    loop = TrainCustomLoop()
    x, t = load_data()
    batch_x = x[1 * BATCH_SIZE: 2 * BATCH_SIZE]
    batch_t = t[1 * BATCH_SIZE: 2 * BATCH_SIZE]
    return loop, x, t, batch_x, batch_t


def test_shuffle_data(setup):
    loop, x, t, _, _ = setup
    xx, tt = loop._shuffle_data(x, t)
    assert xx.shape == (300, 2)
    assert tt.shape == (300, 3)


def test_update_params_with_grads(setup):
    loop, _, _, batch_x, batch_t = setup
    assert loop._update_params_with_grads(
        batch_x, batch_t, 0, 0) == pytest.approx(1.111674016212609, rel=1e-6)


def test_learning_process(setup):
    loop, x, _, batch_x, batch_t = setup
    loss = loop._update_params_with_grads(batch_x, batch_t, 0, 0)
    max_iters = len(x) // BATCH_SIZE
    *_, process = loop._learning_process(loss, 1, 9, 9, max_iters)
    assert process == '| epoch 10 | iter 10 / 10 | loss 1.11'


def test_update(setup):
    loop, x, t, _, _ = setup
    assert len(loop.update(x, t, MAX_EPOCH, BATCH_SIZE)) == 300


def test_save_plot_image(setup):
    loop, x, t, _, _ = setup
    loop.update(x, t, MAX_EPOCH, BATCH_SIZE)
    file_path = '../img/train_custom_loop_plot.png'
    loop.save_plot_image(file_path)
    assert os.path.exists(file_path)


def test_save_dicision_boundary_image(setup):
    loop, x, t, _, _ = setup
    loop.update(x, t, MAX_EPOCH, BATCH_SIZE)
    file_path = '../img/dicision_boundary.png'
    loop.save_dicision_boundary_image(x, t, file_path)
    assert os.path.exists(file_path)
