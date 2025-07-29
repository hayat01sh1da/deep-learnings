import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
from vector import Vector

class TestVector(unittest.TestCase):
    def setUp(self):
        self.vector   = Vector(np.array([1, 2, 3]))
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_get_class_name(self):
        self.assertEqual(str(self.vector.x.__class__), '<class 'numpy.ndarray'>')

    def test_get_shape(self):
        self.assertEqual(self.vector.x.shape, (3,))

    def test_get_dim(self):
        self.assertEqual(self.vector.x.ndim, 1)

    def test_calc_inner_product(self):
        inner_product = self.vector.calc_inner_product(np.array([4, 5, 6]))
        self.assertEqual(inner_product, 32)

if __name__ == '__main__':
    unittest.main()
