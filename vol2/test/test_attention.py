import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from attention import Attention

class TestAttemtion(unittest.TestCase):
    def setUp(self):
        self.attention = Attention()
        self.hs        = np.random.randn(10, 5, 4)
        self.h         = np.random.randn(10, 4)
        self.pycaches  = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.attention.forward(self.hs, self.h)
        self.assertEqual((10, 4), out.shape)

    def test_backward(self):
        dout    = self.attention.forward(self.hs, self.h)
        dhs, dh = self.attention.backward(dout)
        self.assertEqual((10, 5, 4), dhs.shape)
        self.assertEqual((10, 5), dh.shape)

if __name__ == '__main__':
    unittest.main()
