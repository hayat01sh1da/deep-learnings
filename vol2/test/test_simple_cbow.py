import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from count_based_methods import CountBasedMethod
from simple_cbow import SimpleCBOW
from simple_word2vec import SimpleWord2Vec


@pytest.fixture
def setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, _, corpus = cbm.preprocess(word_list)
    vocab_size = len(word_to_id)
    simple_cbow = SimpleCBOW(vocab_size, 3)
    word2vec = SimpleWord2Vec()
    contexts_array, target_array = word2vec.create_contexts_target(corpus)
    contexts = word2vec.convert_to_one_hot(contexts_array, vocab_size)
    target = word2vec.convert_to_one_hot(target_array, vocab_size)
    return simple_cbow, contexts, target


def test_forward(setup):
    simple_cbow, contexts, target = setup
    assert round(simple_cbow.forward(contexts, target), 3) == 1.946


def test_grads_diff(setup):
    simple_cbow, contexts, target = setup
    before_in_grads_0 = copy.copy(simple_cbow.in_layer_0.grads[0])
    before_in_grads_1 = copy.copy(simple_cbow.in_layer_1.grads[0])
    before_out_grads = copy.copy(simple_cbow.out_layer.grads[0])

    simple_cbow.forward(contexts, target)
    simple_cbow.backward()

    after_in_grads_0 = simple_cbow.in_layer_0.grads[0]
    after_in_grads_1 = simple_cbow.in_layer_1.grads[0]
    after_out_grads = simple_cbow.out_layer.grads[0]

    assert_array_equal(
        before_in_grads_0 == after_in_grads_0,
        np.array([
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, True, True],
            [True, True, True],
        ]),
    )
    assert_array_equal(
        before_in_grads_1 == after_in_grads_1,
        np.array([
            [True, True, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]),
    )
    assert_array_equal(
        before_out_grads == after_out_grads,
        np.zeros((3, 7), dtype=bool),
    )
