import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from attention_encoder import AttentionEncoder

class TestAttentionEncoder(unittest.TestCase):
    def setUp(self):
        vocab_size             = 13
        wordvec_size           = 100
        hidden_size            = 100
        self.attention_encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.xs                = np.random.randint(0, 13, (7, 3))
        self.pycaches          = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        hs = self.attention_encoder.forward(self.xs)
        self.assertEqual((7, 3, 100), hs.shape)

    def test_backward(self):
        dhs  = self.attention_encoder.forward(self.xs)
        dout = self.attention_encoder.backward(dhs)
        self.assertEqual(None, dout)

if __name__ == '__main__':
    unittest.main()
