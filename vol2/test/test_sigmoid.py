import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid  = Sigmoid()
        self.x        = np.random.randn(10, 4)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.sigmoid.forward(self.x)
        self.assertEqual(out.shape, (10, 4))

    def test_backward(self):
        self.sigmoid.forward(self.x)
        dout = np.random.randn(10, 4)
        dx   = self.sigmoid.backward(dout)
        self.assertEqual(dx.shape, (10, 4))

if __name__ == '__main__':
    unittest.main()
