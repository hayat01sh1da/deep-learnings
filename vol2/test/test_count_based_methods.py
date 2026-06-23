import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal

from count_based_methods import CountBasedMethod


@pytest.fixture
def cbm_setup():
    cbm = CountBasedMethod()
    word_list = cbm.text_to_word_list('You said good-bye and I said hello.')
    word_to_id, id_to_word, corpus = cbm.preprocess(word_list)
    co_matrix = cbm.create_co_matrix(corpus, len(word_to_id))
    return {
        'cbm': cbm,
        'word_list': word_list,
        'query': 'you',
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'corpus': corpus,
        'vocab_size': len(word_to_id),
        'co_matrix': co_matrix,
    }


def test_words(cbm_setup):
    assert cbm_setup['word_list'] == ['you', 'said',
                                      'good-bye', 'and', 'i', 'said', 'hello', '.']


def test_take_out_query(cbm_setup):
    query_info, query_vec = cbm_setup['cbm']._take_out_query(
        cbm_setup['query'], cbm_setup['word_to_id'], cbm_setup['co_matrix'],
    )
    assert query_info == {'query': 'you'}
    assert_array_equal(query_vec, np.array([0, 1, 0, 0, 0, 0, 0]))


def test_cos_similarity(cbm_setup):
    x = cbm_setup['co_matrix'][cbm_setup['word_to_id']['you']]
    y = cbm_setup['co_matrix'][cbm_setup['word_to_id']['i']]
    assert cbm_setup['cbm']._cos_similarity(x, y) == 0.7071067691154799


def test_calc_cos_similarity(cbm_setup):
    *_, query_vec = cbm_setup['cbm']._take_out_query(
        cbm_setup['query'], cbm_setup['word_to_id'], cbm_setup['co_matrix'],
    )
    similarity = cbm_setup['cbm']._calc_cos_similarity(
        cbm_setup['vocab_size'], cbm_setup['co_matrix'], query_vec,
    )
    assert_almost_equal(
        similarity,
        np.array([1.0, 0.0, 0.7071068, 0.0, 0.7071068, 0.7071068, 0.0]),
    )


def test_output_result_asc(cbm_setup):
    *_, query_vec = cbm_setup['cbm']._take_out_query(
        cbm_setup['query'], cbm_setup['word_to_id'], cbm_setup['co_matrix'],
    )
    similarity = cbm_setup['cbm']._calc_cos_similarity(
        cbm_setup['vocab_size'], cbm_setup['co_matrix'], query_vec,
    )
    assert cbm_setup['cbm']._output_result_asc(
        similarity, cbm_setup['query'], cbm_setup['id_to_word'],
    ) == {
        'good-bye': 0.7071067691154799,
        'i': 0.7071067691154799,
        'hello': 0.7071067691154799,
        'said': 0.0,
        'and': 0.0,
    }


def test_preprocess(cbm_setup):
    assert cbm_setup['word_to_id'] == {
        'you': 0,
        'said': 1,
        'good-bye': 2,
        'and': 3,
        'i': 4,
        'hello': 5,
        '.': 6,
    }
    assert cbm_setup['id_to_word'] == {
        0: 'you',
        1: 'said',
        2: 'good-bye',
        3: 'and',
        4: 'i',
        5: 'hello',
        6: '.',
    }
    assert_array_equal(cbm_setup['corpus'], np.array([0, 1, 2, 3, 4, 1, 5, 6]))


def test_create_co_matrix(cbm_setup):
    assert_array_equal(
        cbm_setup['co_matrix'],
        np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ]),
    )


def test_rank_similarity(cbm_setup):
    assert cbm_setup['cbm'].rank_similarities(
        cbm_setup['query'], cbm_setup['word_to_id'], cbm_setup['co_matrix'],
        cbm_setup['vocab_size'], cbm_setup['id_to_word'],
    ) == {
        'query': 'you',
        'good-bye': 0.7071067691154799,
        'i': 0.7071067691154799,
        'hello': 0.7071067691154799,
        'said': 0.0,
        'and': 0.0,
    }


def test_ppmi(cbm_setup):
    assert_almost_equal(
        cbm_setup['cbm'].ppmi(cbm_setup['co_matrix']),
        np.array([
            [0.0, 1.8073549, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.8073549, 0.0, 0.8073549, 0.0, 0.8073549, 0.8073549, 0.0],
            [0.0, 0.8073549, 0.0, 1.8073549, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8073549, 0.0, 1.8073549, 0.0, 0.0],
            [0.0, 0.8073549, 0.0, 1.8073549, 0.0, 0.0, 0.0],
            [0.0, 0.8073549, 0.0, 0.0, 0.0, 0.0, 2.807355],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.807355, 0.0],
        ]),
    )


def test_svd(cbm_setup):
    M = cbm_setup['cbm'].ppmi(cbm_setup['co_matrix'])
    U = cbm_setup['cbm'].singular_value_deconposition(M)
    # The left singular vectors are not unique: their signs are arbitrary and,
    # because every non-zero singular value here is degenerate (each appears
    # twice), any rotation within the shared subspace is an equally valid U.
    # Pinning exact values is therefore brittle -- it broke on the LAPACK update
    # shipped with numpy 2.5.0. Instead assert the properties that *are*
    # invariant: the (unique) singular values, that U is orthonormal, and that U
    # diagonalises M @ M.T with the squared singular values.
    S = np.linalg.svd(M, compute_uv=False)
    assert U.shape == M.shape
    assert_almost_equal(
        S,
        np.array([3.1680453, 3.1680453, 2.7029872,
                  2.7029872, 1.5144811, 1.5144811, 0.0]),
    )
    eye = np.eye(M.shape[0])
    assert_almost_equal(U @ U.T, eye, decimal=5)
    assert_almost_equal(U @ np.diag(S ** 2) @ U.T, M @ M.T, decimal=5)


def test_save_svd_plot_image(cbm_setup):
    M = cbm_setup['cbm'].ppmi(cbm_setup['co_matrix'])
    U = cbm_setup['cbm'].singular_value_deconposition(M)
    file_path = '../img/svd_plot.png'
    cbm_setup['cbm'].save_svd_plot_image(cbm_setup['word_to_id'], U, file_path)
    assert os.path.exists(file_path)
