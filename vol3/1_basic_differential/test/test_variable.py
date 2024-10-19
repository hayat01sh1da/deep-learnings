import unittest
import sys
import os
import shutil
import glob
sys.path.append('./1_basic_differential/src')
from variable import Variable
import numpy as np
from numpy.testing import assert_array_equal

class TestVariable(unittest.TestCase):
    def setUp(self):
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_variable_1(self):
        data     = np.array(1)
        variable = Variable(data)
        assert_array_equal(np.array(1), variable.get_data())
        self.assertEqual(0, variable.get_data().ndim)
        new_data = np.array([10])
        variable.set_data(new_data)
        assert_array_equal(np.array(10), variable.get_data())

    def test_variable_2(self):
        data     = np.array([1, 2, 3])
        variable = Variable(data)
        assert_array_equal(np.array([1, 2, 3]), variable.get_data())
        self.assertEqual(1, variable.get_data().ndim)
        new_data = np.array([1.0, 2.0, 3.0])
        variable.set_data(new_data)
        assert_array_equal(np.array([1., 2., 3.]), variable.get_data())

    def test_variable_3(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        variable = Variable(data)
        assert_array_equal(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]), variable.get_data())
        self.assertEqual(2, variable.get_data().ndim)
        new_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        variable.set_data(new_data)
        assert_array_equal(np.array([
            [1., 2., 3.],
            [4., 5., 6.]
        ]), variable.get_data())

if __name__ == '__main__':
    unittest.main()
