import unittest
import sys
sys.path.append('./1_basic_differential/src')
from diff import *
from square import Square
from variable import Variable
import numpy as np

class TestDifferenciation(unittest.TestCase):
    def setUp(self):
        data   = np.array(0.5)
        self.x = Variable(data)

    def test_numerical_diff_1(self):
        dy = numerical_diff(f, self.x)
        self.assertEqual(3.2974426293330694, dy)

    def test_f(self):
        dy = f(self.x)
        self.assertEqual(1.648721270700128, dy.data)

if __name__ == '__main__':
    unittest.main()
