import numpy as np
import pytest
from numpy.testing import assert_array_equal

from count_based_methods import CountBasedMethod
from simple_word2vec import SimpleWord2Vec


@pytest.fixture
def setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, _, corpus = cbm.preprocess(word_list)
    word2vec = SimpleWord2Vec()
    contexts_array, target_array = word2vec.create_contexts_target(corpus)
    return word2vec, word_to_id, corpus, contexts_array, target_array


def test_corpus(setup):
    _, _, corpus, _, _ = setup
    assert_array_equal(corpus, np.array([0, 1, 2, 3, 4, 1, 5, 6]))


def test_create_contexts_target(setup):
    _, _, _, contexts_array, target_array = setup
    assert contexts_array.shape == (6, 2)
    assert_array_equal(
        contexts_array,
        np.array([[0, 2], [1, 3], [2, 4], [3, 1], [4, 5], [1, 6]]),
    )
    assert target_array.shape == (6,)
    assert_array_equal(target_array, np.array([1, 2, 3, 4, 1, 5]))


def test_convert_to_one_hot(setup):
    word2vec, word_to_id, _, contexts_array, target_array = setup
    vocab_size = len(word_to_id)
    contexts = word2vec.convert_to_one_hot(contexts_array, vocab_size)
    assert contexts.shape == (6, 2, 7)
    assert_array_equal(
        contexts,
        np.array([
            [[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
            [[0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]],
            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
        ]),
    )
    target = word2vec.convert_to_one_hot(target_array, vocab_size)
    assert target.shape == (6, 7)
    assert_array_equal(
        target,
        np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
        ]),
    )
