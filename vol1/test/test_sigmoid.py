import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid  = Sigmoid()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        x = np.array([[1.0, -0.5], [-2.0, 3.0]])
        assert_almost_equal(([[0.73105858, 0.37754067], [0.11920292, 0.95257413]]), self.sigmoid.forward(x))

    def test_backward(self):
        x = np.array([[1.0, -0.5], [-2.0, 3.0]])
        self.sigmoid.forward(x)
        dout = 1
        assert_almost_equal(np.array([[0.0386563, 0.0552267], [0.0110237, 0.0020409]]), self.sigmoid.backward(self.sigmoid.backward(dout)))

if __name__ == '__main__':
    unittest.main()
