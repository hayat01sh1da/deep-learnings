import unittest
import sys
sys.path.append('./1_basic_differential/src')
from variable import Variable
from square import Square
import numpy as np
from numpy.testing import assert_array_equal

class TestSquare(unittest.TestCase):
    def setUp(self):
        self.square = Square()

    def test_call(self):
        data = np.array([
            [0, 4, 0],
            [0, 2, 5],
            [9, 9, 1]
        ])
        input = Variable(data)
        output = self.square(input)
        assert_array_equal(np.array([
            [0,  16,  0],
            [0,   4, 25],
            [81, 81,  1]
        ]), output.data)

    def test_forward(self):
        x = np.array([
            [0, 4, 0],
            [0, 2, 5],
            [9, 9, 1]
        ])
        y = self.square.forward(x)
        assert_array_equal(np.array([
            [0,  16,  0],
            [0,   4, 25],
            [81, 81,  1]
        ]), y)

if __name__ == '__main__':
    unittest.main()
