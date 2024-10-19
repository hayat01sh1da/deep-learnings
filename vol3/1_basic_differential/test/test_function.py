import unittest
import sys
import os
import shutil
import glob
sys.path.append('./1_basic_differential/src')
from variable import Variable
from function import Function
import numpy as np

class TestFunction(unittest.TestCase):
    def setUp(self):
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)
    def test_call(self):
        data = np.array([
            [0, 4, 0],
            [0, 2, 5],
            [9, 9, 1]
        ])
        input = Variable(data)
        f     = Function()
        with self.assertRaises(NotImplementedError):
            output = f(input)

if __name__ == '__main__':
    unittest.main()
