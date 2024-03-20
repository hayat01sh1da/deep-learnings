import unittest
import sys
sys.path.append('./1_basic_differential/src')
from exp import Exp
from square import Square
from variable import Variable
import numpy as np

class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.input = np.array(0.5)
        self.exp = Exp()

    def test_call(self):
        square_1 = Square()
        square_2 = Square()
        x = Variable(self.input)
        a = square_1(x)
        b = self.exp(a)
        y = square_2(b)
        self.assertEqual(0.25, a.data)
        self.assertEqual(1.2840254166877414, b.data)
        self.assertEqual(1.648721270700128, y.data)

    def test_forward(self):
        y = self.exp.forward(self.input)
        self.assertEqual(1.6487212707001282, y)

if __name__ == '__main__':
    unittest.main()
