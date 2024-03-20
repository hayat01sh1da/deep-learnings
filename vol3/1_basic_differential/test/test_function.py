import unittest
import sys
sys.path.append('./1_basic_differential/src')
from variable import Variable
from function import Function
import numpy as np
from numpy.testing import assert_array_equal

class TestFunction(unittest.TestCase):
    def test_call(self):
        data = np.array([
            [0, 4, 0],
            [0, 2, 5],
            [9, 9, 1]
        ])
        input = Variable(data)
        f = Function()
        with self.assertRaises(NotImplementedError):
            output = f(input)

if __name__ == '__main__':
    unittest.main()
