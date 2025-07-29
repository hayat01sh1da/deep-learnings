import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from decoder import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        vocab_size    = 13
        wordvec_size  = 100
        hidden_size   = 100
        self.decoder  = Decoder(vocab_size, wordvec_size, hidden_size)
        self.xs       = np.random.randint(0, 13, (13, 100))
        self.h        = np.random.randn(13, 100)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        score = self.decoder.forward(self.xs, self.h)
        self.assertEqual(score.shape, (13, 100, 13))

    def test_backward(self):
        dscore = self.decoder.forward(self.xs, self.h)
        dh     = self.decoder.backward(dscore)
        self.assertEqual(dh.shape, (13, 100))

    def test_generate(self):
        h           = np.random.randn(1, 100)
        start_id    = 0
        sample_size = 10
        sampled     = self.decoder.generate(h, start_id, sample_size)
        self.assertEqual(len(sampled), 10)

if __name__ == '__main__':
    unittest.main()
