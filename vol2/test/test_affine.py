import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from affine import Affine

class TestAffine(unittest.TestCase):
    def setUp(self):
        W             = np.random.randn(2, 4)
        b             = np.random.randn(4)
        self.affine   = Affine(W, b)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.affine.forward(np.random.randn(4, 2))
        self.assertEqual((4, 4), out.shape)

    def test_backward(self):
        self.affine.forward(np.random.randn(4, 2))
        dx = self.affine.backward(np.random.randn(4, 4))
        self.assertEqual((4, 2), dx.shape)

if __name__ == '__main__':
    unittest.main()
