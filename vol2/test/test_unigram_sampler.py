import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from unigram_sampler import UnigramSampler


@pytest.fixture
def sampler():
    return UnigramSampler(np.array([0, 1, 2, 3, 4, 1, 2, 3]), 0.75, 2)


def test_word_p(sampler):
    assert_almost_equal(
        sampler.word_p,
        np.array([0.141937, 0.2387087, 0.2387087, 0.2387087, 0.141937]),
    )


def test_get_negative_sample(sampler):
    assert sampler.get_negative_sample(np.array([1, 3, 0])).shape == (3, 2)
