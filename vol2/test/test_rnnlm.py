import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/')
sys.path.append('./src/concerns')
sys.path.append('./src/layers')
from rnnlm import RNNLM
from count_based_methods import CountBasedMethod

class TestRNNLM(unittest.TestCase):
    def setUp(self):
        text           = 'You said good-bye and I said hello.'
        cbm            = CountBasedMethod()
        word_list      = cbm.text_to_word_list(text)
        word_to_id, *_ = cbm.preprocess(word_list)
        vocab_size     = len(word_to_id)
        wordvec_size   = 100
        hidden_size    = 100
        self.rnnlm     = RNNLM(vocab_size, wordvec_size, hidden_size)
        self.xs        = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])
        self.ts = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_predict(self):
        score = self.rnnlm._predict(self.xs)
        self.assertEqual(score.shape, (2, 4, 7))

    def test_forward(self):
        loss = self.rnnlm.forward(self.xs, self.ts)
        self.assertEqual(round(loss, 2), 1.94)

    def test_backward(self):
        self.rnnlm.forward(self.xs, self.ts)
        dout = self.rnnlm.backward()
        self.assertEqual(dout, None)

    def test_reset_state(self):
        self.rnnlm.forward(self.xs, self.ts)
        self.rnnlm.backward()
        self.assertEqual(self.rnnlm.lstm_layer.h.shape, (2, 100))
        self.rnnlm.reset_state()
        self.assertEqual(self.rnnlm.lstm_layer.h, None)

    def test_save_params(self):
        self.rnnlm.forward(self.xs, self.ts)
        self.rnnlm.backward()
        self.rnnlm.save_params()
        self.assertTrue(os.path.exists('../pkl/rnnlm.pkl'))

    def test_load_params(self):
        self.rnnlm.load_params()
        a, b, c, d, e, f = self.rnnlm.params
        self.assertEqual(a.shape, (7, 100))
        self.assertEqual(b.shape, (100, 400))
        self.assertEqual(c.shape, (100, 400))
        self.assertEqual(d.shape, (400,))
        self.assertEqual(e.shape, (100, 7))
        self.assertEqual(f.shape, (7,))

if __name__ == '__main__':
    unittest.main()
