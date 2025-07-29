import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from time_affine import TimeAffine

class TestTimeAffine(unittest.TestCase):
    def setUp(self):
        W                = np.random.randn(3, 3)
        b                = np.random.randn(3,)
        self.time_affine = TimeAffine(W, b)
        self.x           = np.random.randn(3, 3, 3)
        self.pycaches    = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.time_affine.forward(self.x)
        self.assertEqual(out.shape, (3, 3, 3))

    def test_backward(self):
        dout = self.time_affine.forward(self.x)
        dx   = self.time_affine.backward(dout)
        self.assertEqual(dout.shape, (3, 3, 3))

if __name__ == '__main__':
    unittest.main()
