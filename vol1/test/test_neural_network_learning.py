import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append('./src')
from neural_network_learning import NeuralNetworkLearning

class TestNeuralNetworkLearning(unittest.TestCase):
    def setUp(self):
        self.nwl = NeuralNetworkLearning()

    def test_mean_squared_error_1(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        result = self.nwl.mean_squared_error(np.array(y), np.array(t))
        self.assertEqual(0.09750000000000003, result)

    def test_mean_squared_error_2(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        result = self.nwl.mean_squared_error(np.array(y), np.array(t))
        self.assertEqual(0.5975, result)

    def test_cross_entropy_error_1(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        result = self.nwl.cross_entropy_error(np.array(y), np.array(t))
        self.assertEqual(0.510825457099338, result)

    def test_cross_entropy_error_2(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        result = self.nwl.cross_entropy_error(np.array(y), np.array(t))
        self.assertEqual(2.302584092994546, result)

    # y = 0.01 * x ** 2 + 0.1 * x
    def function_1(self, x):
        return 0.01 * x ** 2 + 0.1 * x

    def test_numerical_diff_1(self):
        result1 = self.nwl.numerical_diff(self.function_1, 5)
        self.assertEqual(0.1999999999990898, result1)
        result2 = self.nwl.numerical_diff(self.function_1, 10)
        self.assertEqual(0.2999999999986347, result2)

    # f(x0, x1) = x0 ** 2 + x1 ** 2
    def function_2(self, x):
        return np.sum(x ** 2)

    def test_numerical_diff_2(self):
        result1 = self.nwl.numerical_diff(self.function_2, 5)
        self.assertEqual(9.999999999976694, result1)
        result2 = self.nwl.numerical_diff(self.function_2, 10)
        self.assertEqual(19.99999999995339, result2)

    # Return αf / αx0 to x0 when x0 = 3, x1 = 4
    def function_3(self, x0):
        return x0 * x0 + 4.0 ** 2.0

    def test_numerical_diff_3(self):
        result1 = self.nwl.numerical_diff(self.function_3, 3.0)
        self.assertEqual(6.00000000000378, result1)

    # Return αf / αx1 to x1 when x0 = 3, x1 = 4
    def function_4(self, x1):
        return 3.0 ** 2.0 + x1 * x1

    def test_numerical_diff_4(self):
        result1 = self.nwl.numerical_diff(self.function_4, 4.0)
        self.assertEqual(7.999999999999119, result1)

    def test_numerial_gradient(self):
        result1 = self.nwl.numerial_gradient(self.function_2, np.array([3.0, 4.0]))
        assert_almost_equal(np.array([6., 8.]), result1)
        result2 = self.nwl.numerial_gradient(self.function_2, np.array([0.0, 2.0]))
        assert_almost_equal(np.array([0., 4.]), result2)
        result3 = self.nwl.numerial_gradient(self.function_2, np.array([3.0, 0.0]))
        assert_almost_equal(np.array([6., 0.]), result3)

    def test_gradient_descent(self):
        result = self.nwl.gradient_descent(self.function_2, np.array([3.0, 4.0]), learning_rate=0.1)
        assert_almost_equal(np.array([6.11110793e-10, 8.14814391e-10]), result)

if __name__ == '__main__':
    unittest.main()
