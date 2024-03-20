import unittest
import sys
sys.path.append('./src')
from mul_layer import MulLayer

class TestMulLayer(unittest.TestCase):
    def setUp(self):
        self.apple_layer = MulLayer()
        self.tax_layer   = MulLayer()
        self.apple       = 100
        self.apple_num   = 2
        self.tax         = 1.1

    def test_forward(self):
        apple_price = self.apple_layer.forward(self.apple, self.apple_num)
        price       = self.tax_layer.forward(apple_price, self.tax)
        self.assertEqual(220, int(price))

    def test_backward(self):
        apple_price = self.apple_layer.forward(self.apple, self.apple_num)
        self.tax_layer.forward(apple_price, self.tax)
        dprice             = 1
        dapple_price, dtax = self.tax_layer.backward(dprice)
        dapple, dapple_num = self.apple_layer.backward(dapple_price)
        self.assertEqual(2.2, dapple)
        self.assertEqual(110, int(dapple_num))
        self.assertEqual(200, dtax)

if __name__ == '__main__':
    unittest.main()
