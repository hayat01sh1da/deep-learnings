import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
import os
import shutil
import glob
sys.path.append('./src/concerns')
from unigram_sampler import UnigramSampler

class TestUnigramSampler(unittest.TestCase):
    def setUp(self):
        corpus               = np.array([0, 1, 2, 3, 4, 1, 2, 3])
        power                = 0.75
        sample_size          = 2
        self.unigram_sampler = UnigramSampler(corpus, power, sample_size)
        self.pycaches        = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_word_p(self):
        word_p = self.unigram_sampler.word_p
        assert_almost_equal(np.array(
            [0.141937, 0.2387087, 0.2387087, 0.2387087, 0.141937]
        ), word_p)

    def test_get_negative_sample(self):
        target          = np.array([1, 3, 0])
        negative_sample = self.unigram_sampler.get_negative_sample(target)
        self.assertEqual((3, 2), negative_sample.shape)

if __name__ == '__main__':
    unittest.main()
