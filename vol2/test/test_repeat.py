import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from repeat import Repeat

class TestRepeat(unittest.TestCase):
    def setUp(self):
        self.repeat   = Repeat(8, 7)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        x = np.random.randn(1, self.repeat.D)
        self.assertEqual((7, 8), self.repeat.forward(x).shape)

    def test_backward(self):
        dy = np.random.randn(self.repeat.N, self.repeat.D)
        self.assertEqual((1, 8), self.repeat.backward(dy).shape)

if __name__ == '__main__':
    unittest.main()
