import unittest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal
import os
import shutil
import glob
import sys
sys.path.append('./src')
sys.path.append('./dataset')
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nnw      = NeuralNetwork()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_sigmoid(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y = self.nnw._sigmoid(x)
        assert_almost_equal(np.array(
            [
                0.00669285, 0.00739154, 0.00816257, 0.0090133 , 0.0099518 ,
                0.01098694, 0.01212843, 0.01338692, 0.01477403, 0.0163025 ,
                0.01798621, 0.01984031, 0.02188127, 0.02412702, 0.02659699,
                0.02931223, 0.03229546, 0.03557119, 0.03916572, 0.04310725,
                0.04742587, 0.05215356, 0.05732418, 0.06297336, 0.06913842,
                0.07585818, 0.0831727 , 0.09112296, 0.09975049, 0.10909682,
                0.11920292, 0.13010847, 0.14185106, 0.15446527, 0.16798161,
                0.18242552, 0.19781611, 0.21416502, 0.23147522, 0.24973989,
                0.26894142, 0.2890505 , 0.31002552, 0.33181223, 0.35434369,
                0.37754067, 0.40131234, 0.42555748, 0.450166  , 0.47502081,
                0.5       , 0.52497919, 0.549834  , 0.57444252, 0.59868766,
                0.62245933, 0.64565631, 0.66818777, 0.68997448, 0.7109495 ,
                0.73105858, 0.75026011, 0.76852478, 0.78583498, 0.80218389,
                0.81757448, 0.83201839, 0.84553473, 0.85814894, 0.86989153,
                0.88079708, 0.89090318, 0.90024951, 0.90887704, 0.9168273 ,
                0.92414182, 0.93086158, 0.93702664, 0.94267582, 0.94784644,
                0.95257413, 0.95689275, 0.96083428, 0.96442881, 0.96770454,
                0.97068777, 0.97340301, 0.97587298, 0.97811873, 0.98015969,
                0.98201379, 0.9836975 , 0.98522597, 0.98661308, 0.98787157,
                0.98901306, 0.9900482 , 0.9909867 , 0.99183743, 0.99260846
            ]
        ), y)
        # self.assertEqual(True, os.path.exists('../img/sigmoid.png'))

    def test_softmax(self):
        y = self.nnw._softmax(np.array([0.3, 2.9, 4.0]))
        assert_almost_equal(np.array([0.01821127, 0.24519181, 0.73659691]), y)

    def test_show_image(self):
        x_train, t_train         = self.nnw._get_test_data()
        img, label, reshaped_img = self.nnw._process_image(x_train, t_train)
        self.assertEqual(7, label)
        self.assertEqual((784,), img.shape)
        self.assertEqual((28, 28), reshaped_img.shape)
        # self.nnw._show_image(reshaped_img)

    def test_step_func(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y = self.nnw.step_func(x)
        assert_array_equal(np.array(
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        ), y)
        # self.assertEqual(True, os.path.exists('../img/step_func.png'))

    def test_relu(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y = self.nnw.relu(x)
        assert_almost_equal(np.array(
            [
                0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4,
                1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. ,
                4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9
            ]
        ), y)
        # self.assertEqual(True, os.path.exists('../img/relu.png'))

    def test_matrix_product_1(self):
        a       = np.array([[1, 2], [3, 4]])
        b       = np.array([[5, 6], [7, 8]])
        product = self.nnw.matrix_product(a, b)
        self.assertEqual((2, 2), a.shape)
        self.assertEqual((2, 2), b.shape)
        assert_array_equal(np.array(
            [
                [19, 22],
                [43, 50]
            ]
        ), product)

    def test_matrix_product_2(self):
        a       = np.array([[1, 2, 3], [4, 5, 6]])
        b       = np.array([[1, 2], [3, 4], [5,6]])
        product = self.nnw.matrix_product(a, b)
        self.assertEqual((2, 3), a.shape)
        self.assertEqual((3, 2), b.shape)
        assert_array_equal(np.array(
            [
                [22, 28],
                [49, 64]
            ]
        ), product)

    def test_matrix_product_3(self):
        a       = np.array([[1, 2], [3, 4], [5,6]])
        b       = np.array([7, 8])
        product = self.nnw.matrix_product(a, b)
        self.assertEqual((3, 2), a.shape)
        self.assertEqual((2,), b.shape)
        assert_array_equal(np.array([23, 53, 83]), product)

    def test_evaluate(self):
        self.assertEqual('92.07%', self.nnw.evaluate())

if __name__ == '__main__':
    unittest.main()
