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
from simple_cbow import SimpleCBOW
from simple_word2vec import SimpleWord2Vec
from count_based_methods import CountBasedMethod

class TestSimpleCBOW(unittest.TestCase):
    def setUp(self):
        text                                   = 'You said good-bye and I said hello.'
        cbm                                    = CountBasedMethod()
        word_list                              = cbm.text_to_word_list(text)
        word_to_id, _, self.corpus             = cbm.preprocess(word_list)
        self.vocab_size                        = len(word_to_id)
        hidden_size                            = 3
        self.simple_cbow                       = SimpleCBOW(self.vocab_size, hidden_size)
        self.simple_word2vec                   = SimpleWord2Vec()
        self.contexts_array, self.target_array = self.simple_word2vec.create_contexts_target(self.corpus)
        self.contexts                          = self.simple_word2vec.convert_to_one_hot(self.contexts_array, self.vocab_size)
        self.target                            = self.simple_word2vec.convert_to_one_hot(self.target_array, self.vocab_size)
        self.pycaches                          = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        loss = self.simple_cbow.forward(self.contexts, self.target)
        self.assertEqual(round(loss, 3), 1.946)

    def test_grads_diff(self):
        before_in_grads_0, = self.simple_cbow.in_layer_0.grads
        before_in_grads_0  = copy.copy(before_in_grads_0)
        before_in_grads_1, = self.simple_cbow.in_layer_1.grads
        before_in_grads_1  = copy.copy(before_in_grads_1)
        before_out_grads,  = self.simple_cbow.out_layer.grads
        before_out_grads   = copy.copy(before_out_grads)
        self.simple_cbow.forward(self.contexts, self.target)
        self.simple_cbow.backward()
        after_in_grads_0, = self.simple_cbow.in_layer_0.grads
        after_in_grads_1, = self.simple_cbow.in_layer_1.grads
        after_out_grads,  = self.simple_cbow.out_layer.grads
        in_grads_0        = before_in_grads_0 == after_in_grads_0
        in_grads_1        = before_in_grads_1 == after_in_grads_1
        out_grads         = before_out_grads == after_out_grads
        assert_array_equal(np.array([
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [ True,  True,  True],
            [ True,  True,  True]
        ]), in_grads_0)
        assert_array_equal(np.array([
            [ True,  True,  True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False]
        ]), in_grads_1)
        assert_array_equal(np.array([
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False]
        ]), out_grads)

if __name__ == '__main__':
    unittest.main()
