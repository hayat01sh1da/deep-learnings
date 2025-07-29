import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        x                     = np.random.randn(10, 2)
        W1                    = np.random.randn(2, 4)
        b1                    = np.random.randn(4)
        W2                    = np.random.randn(4, 3)
        b2                    = np.random.randn(3)
        self.neural_network_1 = NeuralNetwork(x, W1, b1)
        self.neural_network_2 = NeuralNetwork(x, W2, b2)
        self.pycaches         = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_get_hidden_layer_dim(self):
        h = self.neural_network_1.get_hidden_layer()
        self.assertEqual(h.shape, (10, 4))

    def test_get_output_layer(self):
        h  = self.neural_network_1.get_hidden_layer()
        dx = self.neural_network_2.get_output_layer(h)
        self.assertEqual(dx.shape, (10, 3))

if __name__ == '__main__':
    unittest.main()
