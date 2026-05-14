import os
import time

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sgd import SGD
from spiral import load_data
from trainer import Trainer
from two_layer_net import TwoLayerNet


@pytest.fixture
def setup():
    model = TwoLayerNet(input_size=2, hidden_size=10, output_size=3)
    trainer = Trainer(model, SGD(lr=1.0))
    x, t = load_data()
    return trainer, x, t, len(x)


def test_shuffle_data(setup):
    trainer, x, t, data_size = setup
    xx, tt = trainer._shuffle_data(data_size, x, t)
    assert np.array(xx).shape == (300, 2)
    assert np.array(tt).shape == (300, 3)


def test_calculate_loss(setup):
    trainer, x, t, data_size = setup
    batch_size = 32
    xx, tt = trainer._shuffle_data(data_size, x, t)
    batch_x = xx[1 * batch_size: 2 * batch_size]
    batch_t = tt[1 * batch_size: 2 * batch_size]
    assert round(trainer._calculate_loss(batch_x, batch_t), 1) == 1.1


def test_remove_duplicate(setup):
    trainer, x, t, data_size = setup
    batch_size = 32
    xx, tt = trainer._shuffle_data(data_size, x, t)
    batch_x = xx[1 * batch_size: 2 * batch_size]
    batch_t = tt[1 * batch_size: 2 * batch_size]
    trainer._calculate_loss(batch_x, batch_t)
    params, grads = trainer._remove_duplicate()
    param_1, param_2, param_3, param_4 = params
    grad_1, grad_2, grad_3, grad_4 = grads
    assert param_1.shape == (2, 10)
    assert param_2.shape == (10,)
    assert param_3.shape == (10, 3)
    assert param_4.shape == (3,)
    assert grad_1.shape == (2, 10)
    assert grad_2.shape == (10,)
    assert grad_3.shape == (10, 3)
    assert grad_4.shape == (3,)


def test_evaluate(setup):
    trainer, x, t, data_size = setup
    batch_size = 32
    xx, tt = trainer._shuffle_data(data_size, x, t)
    batch_x = xx[1 * batch_size: 2 * batch_size]
    batch_t = tt[1 * batch_size: 2 * batch_size]
    total_loss = trainer._calculate_loss(batch_x, batch_t)
    trainer._remove_duplicate()
    max_iters = data_size // batch_size
    average_loss, training_status = trainer._evaluate(
        total_loss, 1, time.time(), 0, 1, max_iters,
    )
    assert average_loss == 1.0982153338384055
    assert training_status == '| epoch 1 |  iter 2 / 9 | time 0[s] | loss 1.10'


def test_fit(setup):
    trainer, x, t, _ = setup
    training_process = trainer.fit(x, t)
    assert_array_equal(
        np.array(trainer.loss_list),
        np.array([
            1.095670688853155, 1.166632183444487, 1.1176750111873825, 1.1790605366611793,
            1.1308071491715788, 1.1796679902482832, 1.1401573329955508, 1.1263507954471639,
            1.1260232915246664, 1.1018855740540143,
        ]),
    )
    assert training_process == [
        '| epoch 1 |  iter 1 / 9 | time 0[s] | loss 1.10',
        '| epoch 2 |  iter 1 / 9 | time 0[s] | loss 1.17',
        '| epoch 3 |  iter 1 / 9 | time 0[s] | loss 1.12',
        '| epoch 4 |  iter 1 / 9 | time 0[s] | loss 1.18',
        '| epoch 5 |  iter 1 / 9 | time 0[s] | loss 1.13',
        '| epoch 6 |  iter 1 / 9 | time 0[s] | loss 1.18',
        '| epoch 7 |  iter 1 / 9 | time 0[s] | loss 1.14',
        '| epoch 8 |  iter 1 / 9 | time 0[s] | loss 1.13',
        '| epoch 9 |  iter 1 / 9 | time 0[s] | loss 1.13',
        '| epoch 10 |  iter 1 / 9 | time 0[s] | loss 1.10',
    ]


def test_save_plot_image(setup):
    trainer, x, t, _ = setup
    trainer.fit(x, t, max_epoch=300, batch_size=30)
    file_path = '../img/training_plot.png'
    trainer.save_plot_image(file_path)
    assert os.path.exists(file_path)
