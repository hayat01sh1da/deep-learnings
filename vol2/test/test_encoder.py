import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from encoder import Encoder

class TestEncoder(unittest.TestCase):
    def setUp(self):
        vocab_size    = 13
        wordvec_size  = 100
        hidden_size   = 100
        self.encoder  = Encoder(vocab_size, wordvec_size, hidden_size)
        self.xs       = np.random.randint(0, 13, (7, 3))
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        a, b, c, d, e, f, g = self.encoder.forward(self.xs)
        self.assertEqual((100,), a.shape)
        self.assertEqual((100,), b.shape)
        self.assertEqual((100,), c.shape)
        self.assertEqual((100,), d.shape)
        self.assertEqual((100,), e.shape)
        self.assertEqual((100,), f.shape)
        self.assertEqual((100,), g.shape)

    def test_backward(self):
        dh   = self.encoder.forward(self.xs)
        dout = self.encoder.backward(dh)
        self.assertEqual(None, dout)

if __name__ == '__main__':
    unittest.main()
