import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from relu import Relu

class TestRelu(unittest.TestCase):
    def setUp(self):
        self.relu     = Relu()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        x = np.array([[1.0, -0.5], [-2.0, 3.0]])
        assert_array_equal(np.array([[1., 0.], [0., 3.]]), self.relu.forward(x))

    def test_backward(self):
        x = np.array([[1.0, -0.5], [-2.0, 3.0]])
        assert_array_equal(np.array([[1., 0.], [0., 3.]]), self.relu.backward(self.relu.forward(x)))

if __name__ == '__main__':
    unittest.main()
