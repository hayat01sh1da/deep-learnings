import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from matmul import MatMul

class TestMatMul(unittest.TestCase):
    def setUp(self):
        W             = np.random.rand(4, 2)
        self.matmul   = MatMul(W)
        self.x        = np.random.rand(2, 4)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.matmul.forward(self.x)
        self.assertEqual(out.shape, (2, 2))

    def test_backward(self):
        dout = self.matmul.forward(self.x)
        dx   = self.matmul.backward(dout)
        self.assertEqual(dx.shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
