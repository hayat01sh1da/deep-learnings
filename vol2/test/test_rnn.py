import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from rnn import RNN

class TestRNN(unittest.TestCase):
    def setUp(self):
        Wx            = np.random.randn(3, 3)
        Wh            = np.random.randn(7, 3)
        b             = np.random.randn(3,)
        self.rnn      = RNN(Wx, Wh, b)
        self.x        = np.random.randn(7, 3)
        self.h_prev   = np.random.randn(7, 7)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        h_next = self.rnn.forward(self.x, self.h_prev)
        self.assertEqual((7, 3), h_next.shape)

    def test_backward(self):
        h_next      = self.rnn.forward(self.x, self.h_prev)
        dx, dh_prev = self.rnn.backward(h_next)
        self.assertEqual((7, 3), dx.shape)
        self.assertEqual((7, 7), dh_prev.shape)

if __name__ == '__main__':
    unittest.main()
