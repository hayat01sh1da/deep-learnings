import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
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
        # Use isinstance check instead of stringifying the class
        self.assertIsInstance(self.vector.x, np.ndarray)

    def test_get_shape(self):
        self.assertEqual(self.vector.x.shape, (3,))

    def test_get_dim(self):
        self.assertEqual(self.vector.x.ndim, 1)

    def test_calc_inner_product(self):
        inner_product = self.vector.calc_inner_product(np.array([4, 5, 6]))
        self.assertEqual(inner_product, 32)

if __name__ == '__main__':
    unittest.main()
