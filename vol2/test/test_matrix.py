import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from matrix import Matrix

class TestVector(unittest.TestCase):
    def setUp(self):
        self.matrix   = Matrix(np.array([[1, 2, 3], [4, 5, 6]]))
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_get_class_name(self):
        self.assertEqual(str(self.matrix.W.__class__), "<class 'numpy.ndarray'>")

    def test_get_shape(self):
        self.assertEqual(self.matrix.W.shape, (2, 3))

    def test_get_dim(self):
        self.assertEqual(self.matrix.W.ndim, 2)

    def test_calc_sum(self):
        sum = self.matrix.calc_sum(np.array([[0, 1 ,2], [3, 4, 5]]))
        assert_array_equal(np.array([
            [1, 3, 5],
            [7, 9, 11]
        ]), sum)

    def test_calc_product(self):
        calc_product = self.matrix.calc_product(np.array([[0, 1 ,2], [3, 4, 5]]))
        assert_array_equal(np.array([
            [0, 2, 6],
            [12, 20, 30]
        ]), calc_product)

    def test_calc_scala_broadcast(self):
        scala_broadcast = self.matrix.calc_scala_broadcast(10)
        assert_array_equal(np.array([
            [10, 20, 30],
            [40, 50, 60]
        ]), scala_broadcast)

    def test_calc_array_broadcast(self):
        array_broadcast = self.matrix.calc_array_broadcast(np.array([10, 20, 30]))
        assert_array_equal(np.array([
            [10, 40, 90],
            [40, 100, 180]
        ]), array_broadcast)

    def test_calc_inner_product(self):
        inner_product = self.matrix.calc_inner_product(np.array([[7, 8], [9, 10], [11, 12]]))
        assert_array_equal(np.array([
            [58, 64],
            [139, 154]
        ]), inner_product)

if __name__ == '__main__':
    unittest.main()
