import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from weight_sum import WeightSum

class TestWeightSum(unittest.TestCase):
    def setUp(self):
        self.weight_sum = WeightSum()
        self.hs         = np.random.randn(10, 5, 4)
        self.a          = np.random.randn(10, 5)
        self.pycaches   = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        c = self.weight_sum.forward(self.hs, self.a)
        self.assertEqual((10, 4), c.shape)

    def test_backward(self):
        dc      = self.weight_sum.forward(self.hs, self.a)
        dhs, da = self.weight_sum.backward(dc)
        self.assertEqual((10, 5, 4), dhs.shape)
        self.assertEqual((10, 5), da.shape)

if __name__ == '__main__':
    unittest.main()
