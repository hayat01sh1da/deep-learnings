import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from sum import Sum

class TestSum(unittest.TestCase):
    def setUp(self):
        self.sum      = Sum(8, 7)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        x = np.random.randn(self.sum.N, self.sum.D)
        self.assertEqual(self.sum.forward(x).shape, (1, 8))

    def test_backward(self):
        dy = np.random.randn(1, self.sum.D)
        self.assertEqual(self.sum.backward(dy).shape, (7, 8))

if __name__ == '__main__':
    unittest.main()
