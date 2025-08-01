import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from time_rnn import TimeRNN

class TestTimeRNN(unittest.TestCase):
    def setUp(self):
        Wx            = np.random.randn(3, 3)
        Wh            = np.random.randn(3, 3)
        b             = np.random.randn(3,)
        self.time_rnn = TimeRNN(Wx, Wh, b)
        self.xs       = np.random.randn(3, 3, 3)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_state(self):
        h = np.random.randn(7, 7)
        self.time_rnn.set_state(h)
        assert_array_equal(h, self.time_rnn.h)
        self.time_rnn.reset_state()
        self.assertEqual(self.time_rnn.h, None)

    def test_forward(self):
        hs = self.time_rnn.forward(self.xs)
        self.assertEqual(hs.shape, (3, 3, 3))

    def test_backward(self):
        hs  = self.time_rnn.forward(self.xs)
        dxs = self.time_rnn.backward(hs)
        self.assertEqual(dxs.shape, (3, 3, 3))

if __name__ == '__main__':
    unittest.main()
