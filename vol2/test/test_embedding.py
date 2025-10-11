import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from embedding import Embedding

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        W              = np.arange(21).reshape(7, 3)
        self.embedding = Embedding(W)
        self.index     = np.array([0, 2, 0, 4])
        self.pycaches  = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_params(self):
        params, = self.embedding.params
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]
        ]), params)

    def test_grads(self):
        grads, = self.embedding.grads
        assert_array_equal(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]), grads)

    def test_forward(self):
        out = self.embedding.forward(self.index)
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 6,  7,  8],
            [ 0,  1,  2],
            [12, 13, 14]
        ]), out)

    def test_backward(self):
        dout = self.embedding.forward(self.index)
        self.embedding.backward(dout)
        grads, = self.embedding.grads
        assert_array_equal(np.array([
            [ 0,  2,  4],
            [ 0,  0,  0],
            [ 6,  7,  8],
            [ 0,  0,  0],
            [12, 13, 14],
            [ 0,  0,  0],
            [ 0,  0,  0]
        ]), grads)

if __name__ == '__main__':
    unittest.main()
