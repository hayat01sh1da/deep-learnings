import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from peeky_decoder import PeekyDecoder

class TestPeekyDecoder(unittest.TestCase):
    def setUp(self):
        vocab_size         = 13
        wordvec_size       = 16
        hidden_size        = 128
        self.peeky_decoder = PeekyDecoder(vocab_size, wordvec_size, hidden_size)
        self.xs            = np.random.randint(0, 13, (13, 16))
        self.h             = np.random.randn(13, 128)
        self.pycaches      = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        score = self.peeky_decoder.forward(self.xs, self.h)
        self.assertEqual(score.shape, (13, 16, 13))

    def test_backward(self):
        dscore = self.peeky_decoder.forward(self.xs, self.h)
        dh     = self.peeky_decoder.backward(dscore)
        self.assertEqual(dh.shape, (13, 128))

    def test_generate(self):
        h           = np.random.randn(1, 128)
        start_id    = 0
        sample_size = 10
        sampled     = self.peeky_decoder.generate(h, start_id, sample_size)
        self.assertEqual(len(sampled), 10)

if __name__ == '__main__':
    unittest.main()
