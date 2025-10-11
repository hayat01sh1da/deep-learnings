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
from better_rnnlm import BetterRNNLM
from count_based_methods import CountBasedMethod

class TestBetterRNNLM(unittest.TestCase):
    def setUp(self):
        text              = 'You said good-bye and I said hello.'
        cbm               = CountBasedMethod()
        word_list         = cbm.text_to_word_list(text)
        word_to_id, *_    = cbm.preprocess(word_list)
        vocab_size        = len(word_to_id)
        wordvec_size      = 100
        hidden_size       = 100
        self.better_rnnlm = BetterRNNLM(vocab_size, wordvec_size, hidden_size)
        self.xs           = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])
        self.ts = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        import glob, os
        self.pycaches  = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)
#         self.file_path = '../pkl/better_rnnlm.pkl'
        # self.pycaches  = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_predict(self):
        score = self.better_rnnlm._predict(self.xs)
        self.assertEqual(score.shape, (2, 4, 7))

    def test_forward(self):
        loss = self.better_rnnlm.forward(self.xs, self.ts)
        self.assertEqual(round(loss, 2), 1.95)

    def test_backward(self):
        self.better_rnnlm.forward(self.xs, self.ts)
        dout = self.better_rnnlm.backward()
        self.assertEqual(dout, None)

    def test_reset_state(self):
        self.better_rnnlm.forward(self.xs, self.ts)
        self.better_rnnlm.backward()
        self.assertEqual(self.better_rnnlm.lstm_layers[0].h.shape, (2, 100))
        self.better_rnnlm.reset_state()
        self.assertEqual(self.better_rnnlm.lstm_layers[0].h, None)

    # def test_save_params(self):
    #     self.better_rnnlm.forward(self.xs, self.ts)
    #     self.better_rnnlm.backward()
    #     self.better_rnnlm.save_params(self.file_path)
    #     self.assertTrue(os.path.exists(self.file_path))
    #
    # def test_load_params(self):
    #     self.better_rnnlm.load_params(self.file_path)
    #     a, b, c, d, e, f, g, h, i = self.better_rnnlm.params
    #     self.assertEqual(a.shape, (7, 100))
    #     self.assertEqual(b.shape, (100, 400))
    #     self.assertEqual(c.shape, (100, 400))
    #     self.assertEqual(d.shape, (400,))
    #     self.assertEqual(e.shape, (100, 400))
    #     self.assertEqual(f.shape, (100, 400))
    #     self.assertEqual(g.shape, (400,))
    #     self.assertEqual(h.shape, (100, 7))
    #     self.assertEqual(i.shape, (7,))

if __name__ == '__main__':
    unittest.main()
