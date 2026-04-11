import unittest
import sys
import os
import shutil
import glob
sys.path.append('./1_basic_differential/src')
from exp import Exp
from square import Square
from variable import Variable
import numpy as np

class TestTemplate(unittest.TestCase):
    def setUp(self) -> None:
        self.input    = np.array(0.5)
        self.exp: Exp      = Exp()
        self.pycaches: list[str] = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self) -> None:
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_call(self) -> None:
        square_1 = Square()
        square_2 = Square()
        x        = Variable(self.input)
        a        = square_1(x)
        b        = self.exp(a)
        y        = square_2(b)
        self.assertEqual(a.data, 0.25)
        self.assertEqual(b.data, 1.2840254166877414)
        self.assertEqual(y.data, 1.648721270700128)

    def test_forward(self) -> None:
        y = self.exp.forward(self.input)
        self.assertEqual(y, 1.6487212707001282)

if __name__ == '__main__':
    unittest.main()
