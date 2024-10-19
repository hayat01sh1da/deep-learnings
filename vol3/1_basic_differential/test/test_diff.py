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
    def setUp(self):
        data          = np.array(0.5)
        self.x        = Variable(data)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_numerical_diff_1(self):
        dy = numerical_diff(f, self.x)
        self.assertEqual(3.2974426293330694, dy)

    def test_f(self):
        dy = f(self.x)
        self.assertEqual(1.648721270700128, dy.data)

if __name__ == '__main__':
    unittest.main()
