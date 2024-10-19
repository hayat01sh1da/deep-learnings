import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from affine import Affine

class TestAffine(unittest.TestCase):
    def setUp(self):
        W             = np.array([[-0.22472106, -0.42868683, 0.21713442],[-0.13635294, 0.45327181, -1.31839392]])
        b             = np.array([1.55270156, 1.44441689, -1.69451485])
        self.affine   = Affine(W, b)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        x = np.array([1.52949391, -0.81788271])
        assert_almost_equal(([1.32051278,  0.41801982, -0.28411748]), self.affine.forward(x))

    def test_backward(self):
        x = np.array([1.52949391, -0.81788271])
        self.affine.forward(x)
        dout = 1
        assert_almost_equal(np.array([
            [-0.2247211, -0.1363529],
            [-0.4286868,  0.4532718],
            [ 0.2171344, -1.3183939]
        ]), self.affine.backward(dout))

if __name__ == '__main__':
    unittest.main()
