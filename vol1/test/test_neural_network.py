import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys

sys.path.append('./src')

from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nnw = NeuralNetwork()

    def test_sigmoid(self):
        x = np.array([-1.0, 0.0, 1.0])
        y = self.nnw._sigmoid(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        assert_almost_equal(expected, y)

    def test_softmax(self):
        a = np.array([0.3, 2.9, 4.0])
        y = self.nnw._softmax(a)
        assert_almost_equal(np.array([0.01821127, 0.24519181, 0.73659691]), y)

    def test_step_func_and_relu_and_matrix(self):
        x = np.array([-1.0, 0.0, 1.0])
        step = self.nnw.step_func(x)
        assert_array_equal(step, np.array([0, 0, 1]))

        relu = self.nnw.relu(x)
        assert_array_equal(relu, np.array([0.0, 0.0, 1.0]))

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        prod = self.nnw.matrix_product(a, b)
        assert_array_equal(prod, np.array([[19, 22], [43, 50]]))


if __name__ == '__main__':
    import unittest
    import numpy as np
    from numpy.testing import assert_array_equal, assert_almost_equal
    import sys
    import os

    # Ensure src is on the path (relative to this test file)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

    from neural_network import NeuralNetwork


    class TestNeuralNetwork(unittest.TestCase):
        def setUp(self):
            self.nnw = NeuralNetwork()

        def test_sigmoid(self):
            x = np.array([-1.0, 0.0, 1.0])
            y = self.nnw._sigmoid(x)
            expected = 1.0 / (1.0 + np.exp(-x))
            assert_almost_equal(expected, y)

        def test_softmax(self):
            a = np.array([0.3, 2.9, 4.0])
            y = self.nnw._softmax(a)
            assert_almost_equal(np.array([0.01821127, 0.24519181, 0.73659691]), y)

        def test_step_relu_matrix(self):
            x = np.array([-1.0, 0.0, 1.0])
            step = self.nnw.step_func(x)
            assert_array_equal(step, np.array([0, 0, 1]))

            relu = self.nnw.relu(x)
            assert_array_equal(relu, np.array([0.0, 0.0, 1.0]))

            a = np.array([[1, 2], [3, 4]])
            b = np.array([[5, 6], [7, 8]])
            prod = self.nnw.matrix_product(a, b)
            assert_array_equal(prod, np.array([[19, 22], [43, 50]]))


    if __name__ == '__main__':
        unittest.main()
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
import os

# Ensure src is on the path (relative to this test file)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nnw = NeuralNetwork()

    def test_sigmoid(self):
        x = np.array([-1.0, 0.0, 1.0])
        y = self.nnw._sigmoid(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        assert_almost_equal(expected, y)

    def test_softmax(self):
        a = np.array([0.3, 2.9, 4.0])
        y = self.nnw._softmax(a)
        assert_almost_equal(np.array([0.01821127, 0.24519181, 0.73659691]), y)

    def test_step_relu_matrix(self):
        x = np.array([-1.0, 0.0, 1.0])
        step = self.nnw.step_func(x)
        assert_array_equal(step, np.array([0, 0, 1]))

        relu = self.nnw.relu(x)
        assert_array_equal(relu, np.array([0.0, 0.0, 1.0]))

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        prod = self.nnw.matrix_product(a, b)
        assert_array_equal(prod, np.array([[19, 22], [43, 50]]))

if __name__ == '__main__':
    unittest.main()
