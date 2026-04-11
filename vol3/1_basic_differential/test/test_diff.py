import unittest
import sys
sys.path.append('./1_basic_differential/src')
import os
import shutil
import glob
from diff import *
from variable import Variable
import numpy as np

class TestDifferenciation(unittest.TestCase):
    def setUp(self) -> None:
        data          = np.array(0.5)
        self.x: Variable        = Variable(data)
        self.pycaches: list[str] = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self) -> None:
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_numerical_diff_1(self) -> None:
        dy = numerical_diff(f, self.x)
        self.assertEqual(dy, 3.2974426293330694)

    def test_f(self) -> None:
        dy = f(self.x)
        self.assertEqual(dy.data, 1.648721270700128)

if __name__ == '__main__':
    unittest.main()
