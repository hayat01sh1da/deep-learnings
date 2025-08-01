import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
from ptb import *
from rnnlm_gen import RNNLMGen

class TestRNNLMGen(unittest.TestCase):
    def setUp(self):
        corpus, self.word_to_id, self.id_to_word = load_data('train')
        self.rnnlm_gen                           = RNNLMGen()
        self.rnnlm_gen.load_params('../pkl/better_rnnlm.pkl')
        start_word    = 'you'
        self.start_id = self.word_to_id[start_word]
        skip_words    = ['N', '<unk>', '$']
        self.skip_ids = [self.word_to_id[w] for w in skip_words]
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_word_ids_list(self):
        word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
        self.assertEqual(len(word_ids), 100)

    def test_generate_text(self):
        word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
        text     = self.rnnlm_gen.generate_text(self.id_to_word, word_ids)
        self.assertTrue(750 < len(text) < 870)

if __name__ == '__main__':
    unittest.main()
