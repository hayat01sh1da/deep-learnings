import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
sys.path.append('./src/models')
sys.path.append('./src/layers')
from seq2seq import Seq2Seq

class TestSeq2Seq(unittest.TestCase):
    def setUp(self):
        vocab_size    = 13
        wordvec_size  = 100
        hidden_size   = 100
        self.seq2seq  = Seq2Seq(vocab_size, wordvec_size, hidden_size)
        self.xs       = np.random.randint(0, 13, (13, 100))
        self.ts       = np.random.randint(0, 13, (13, 100))
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        loss = self.seq2seq.forward(self.xs, self.ts)
        self.assertEqual(True, 2.55 < round(loss, 2) < 2.58)

    def test_backward(self):
        self.seq2seq.forward(self.xs, self.ts)
        dout = self.seq2seq.backward()
        self.assertEqual(None, dout)

    def test_generate(self):
        xs          = np.random.randint(0, 13, (1, 100))
        start_id    = 0
        sample_size = 10
        sampled = self.seq2seq.generate(xs, start_id, sample_size)
        self.assertEqual(10, len(sampled))

if __name__ == '__main__':
    unittest.main()
