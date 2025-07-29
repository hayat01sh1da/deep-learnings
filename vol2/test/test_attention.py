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
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.attention.forward(self.hs, self.h)
        self.assertEqual(out.shape, (10, 4))

    def test_backward(self):
        dout    = self.attention.forward(self.hs, self.h)
        dhs, dh = self.attention.backward(dout)
        self.assertEqual(dhs.shape, (10, 5, 4))
        self.assertEqual(dh.shape, (10, 5))

if __name__ == '__main__':
    unittest.main()
