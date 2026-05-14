import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cbow import CBOW
from count_based_methods import CountBasedMethod
from simple_word2vec import SimpleWord2Vec


@pytest.fixture
def cbow_setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, _, corpus = cbm.preprocess(word_list)
    cbow = CBOW(len(word_to_id), 2, 1, corpus)
    contexts, target = SimpleWord2Vec().create_contexts_target(corpus)
    return cbow, contexts, target


def test_forward(cbow_setup):
    cbow, contexts, target = cbow_setup
    assert round(cbow.forward(contexts, target), 3) == 4.159


def test_grads_diff(cbow_setup):
    cbow, contexts, target = cbow_setup
    in_layer, *_ = cbow.in_layers
    before_in_layer_grad, = in_layer.grads
    before_in_layer_grad = copy.copy(before_in_layer_grad)
    before_ns_loss_layer_grad, *_ = cbow.ns_loss_layer.grads
    before_ns_loss_layer_grad = copy.copy(before_ns_loss_layer_grad)

    cbow.forward(contexts, target)
    cbow.backward()

    in_layer, *_ = cbow.in_layers
    after_in_layer_grad, = in_layer.grads
    after_ns_loss_layer_grad, *_ = cbow.ns_loss_layer.grads

    assert_array_equal(
        before_in_layer_grad == after_in_layer_grad,
        np.array([
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [True, True],
            [True, True],
        ]),
    )
    assert_array_equal(
        before_ns_loss_layer_grad == after_ns_loss_layer_grad,
        np.array([
            [True, True],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [True, True],
        ]),
    )
