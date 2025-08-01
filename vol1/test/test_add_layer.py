import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src')
from add_layer import AddLayer
from mul_layer import MulLayer

class TestAddLayer(unittest.TestCase):
    def setUp(self):
        self.apple_layer        = MulLayer()
        self.orange_layer       = MulLayer()
        self.apple_orange_layer = AddLayer()
        self.tax_layer          = MulLayer()
        self.apple              = 100
        self.apple_num          = 2
        self.orange             = 150
        self.orange_num         = 3
        self.tax                = 1.1
        self.pycaches           = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        apple_price        = self.apple_layer.forward(self.apple, self.apple_num)
        orange_price       = self.orange_layer.forward(self.orange, self.orange_num)
        apple_orange_price = self.apple_orange_layer.forward(apple_price, orange_price)
        price              = self.tax_layer.forward(apple_orange_price, self.tax)
        self.assertEqual(int(price), 715)

    def test_backward(self):
        apple_price        = self.apple_layer.forward(self.apple, self.apple_num)
        orange_price       = self.orange_layer.forward(self.orange, self.orange_num)
        apple_orange_price = self.apple_orange_layer.forward(apple_price, orange_price)
        self.tax_layer.forward(apple_orange_price, self.tax)
        dprice                      = 1
        dall_price, dtax            = self.tax_layer.backward(dprice)
        dapple_price, dorange_price = self.apple_orange_layer.backward(dall_price)
        dorange, dorange_num        = self.orange_layer.backward(dorange_price)
        dapple, dapple_num          = self.apple_layer.backward(dapple_price)
        self.assertEqual(dapple, 2.2)
        self.assertEqual(int(dapple_num), 110)
        self.assertEqual(float(f'{dorange:.1f}'), 3.3)
        self.assertEqual(int(dorange_num), 165)
        self.assertEqual(dtax, 650)

if __name__ == '__main__':
    unittest.main()
