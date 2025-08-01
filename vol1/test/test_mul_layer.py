import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src')
from mul_layer import MulLayer

class TestMulLayer(unittest.TestCase):
    def setUp(self):
        self.apple_layer = MulLayer()
        self.tax_layer   = MulLayer()
        self.apple       = 100
        self.apple_num   = 2
        self.tax         = 1.1
        self.pycaches    = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        apple_price = self.apple_layer.forward(self.apple, self.apple_num)
        price       = self.tax_layer.forward(apple_price, self.tax)
        self.assertEqual(int(price), 220)

    def test_backward(self):
        apple_price = self.apple_layer.forward(self.apple, self.apple_num)
        self.tax_layer.forward(apple_price, self.tax)
        dprice             = 1
        dapple_price, dtax = self.tax_layer.backward(dprice)
        dapple, dapple_num = self.apple_layer.backward(dapple_price)
        self.assertEqual(dapple, 2.2)
        self.assertEqual(int(dapple_num), 110)
        self.assertEqual(dtax, 200)

if __name__ == '__main__':
    unittest.main()
