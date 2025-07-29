import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
sys.path.append('./src/layers')
sys.path.append('./src/models')
from two_layer_net import TwoLayerNet

class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.two_layer_net = TwoLayerNet(2, 4, 3)
        self.x             = np.random.randn(4, 2)
        self.t             = np.random.randn(4, 3)
        self.pycaches      = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_predict(self):
        x = self.two_layer_net._predict(self.x)
        self.assertEqual(x.shape, (4, 3))

    def test_forward(self):
        loss = self.two_layer_net.forward(self.x, self.t)
        self.assertEqual(int(loss), 1)

    def test_backward(self):
        self.two_layer_net.forward(self.x, self.t)
        dout = self.two_layer_net.backward()
        self.assertEqual(dout.shape, (4, 2))

if __name__ == '__main__':
    unittest.main()
