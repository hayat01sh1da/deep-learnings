import unittest
import numpy as np
from numpy.testing import assert_array_equal
import copy
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
sys.path.append('./src/layers')
sys.path.append('./src/models')
from cbow import CBOW
from simple_word2vec import SimpleWord2Vec
from count_based_methods import CountBasedMethod

class TestSimpleCBOW(unittest.TestCase):
    def setUp(self):
        text                       = 'You said good-bye and I said hello.'
        cbm                        = CountBasedMethod()
        word_list                  = cbm.text_to_word_list(text)
        word_to_id, _, self.corpus = cbm.preprocess(word_list)
        vocab_size                 = len(word_to_id)
        hidden_size                = 2
        window_size                = 1
        self.cbow                  = CBOW(vocab_size, hidden_size, window_size, self.corpus)
        self.simple_word2vec       = SimpleWord2Vec()
        self.contexts, self.target = self.simple_word2vec.create_contexts_target(self.corpus)
        self.pycaches              = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        loss = self.cbow.forward(self.contexts, self.target)
        self.assertEqual(round(loss, 3), 4.159)

    def test_grads_diff(self):
        in_layer, *_                  = self.cbow.in_layers
        before_in_layer_grad,         = in_layer.grads
        before_in_layer_grad          = copy.copy(before_in_layer_grad)
        before_ns_loss_layer_grad, *_ = self.cbow.ns_loss_layer.grads
        before_ns_loss_layer_grad     = copy.copy(before_ns_loss_layer_grad)
        self.cbow.forward(self.contexts, self.target)
        self.cbow.backward()
        in_layer, *_                 = self.cbow.in_layers
        after_in_layer_grad,         = in_layer.grads
        after_ns_loss_layer_grad, *_ = self.cbow.ns_loss_layer.grads
        in_layer_grad                = before_in_layer_grad == after_in_layer_grad
        ns_loss_layer_grad           = before_ns_loss_layer_grad == after_ns_loss_layer_grad
        assert_array_equal(np.array([
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [ True,  True],
            [ True,  True]
        ]), in_layer_grad)
        assert_array_equal(np.array([
            [ True,  True],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [ True,  True]
        ]), ns_loss_layer_grad)

if __name__ == '__main__':
    unittest.main()
