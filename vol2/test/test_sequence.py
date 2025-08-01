import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src/concerns')
from sequence import Sequence

class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence  = Sequence()
        self.file_path = '../texts/addition.txt'
        self.questions = []
        self.answers   = []
        self.pycaches  = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_text_to_dict(self):
        questions, answers = self.sequence._text_to_dict(self.file_path, self.questions, self.answers)
        self.assertEqual(len(questions), 50000)
        self.assertEqual(len(answers), 50000)

    def test_update_vocab(self):
        text = '16+75  _91'
        self.sequence._update_vocab(text)
        self.assertEqual({
            '1': 0,
            '6': 1,
            '+': 2,
            '7': 3,
            '5': 4,
            ' ': 5,
            '_': 6,
            '9': 7
        }, self.sequence.char_to_id)
        self.assertEqual({
            0: '1',
            1: '6',
            2: '+',
            3: '7',
            4: '5',
            5: ' ',
            6: '_',
            7: '9'
        }, self.sequence.id_to_char)

    def test_create_numpy_array(self):
        questions, answers = self.sequence._text_to_dict(self.file_path, self.questions, self.answers)
        self.sequence._create_vocab_dict(questions, answers)
        x, t = self.sequence._create_numpy_array(questions, answers)
        self.assertEqual(x.shape, (50000, 7))
        self.assertEqual(t.shape, (50000, 5))

    def test_load_data(self):
        (x_train, t_train), (x_test, t_test) = self.sequence.load_data(self.file_path)
        self.assertEqual(x_train.shape, (45000, 7))
        self.assertEqual(t_train.shape, (45000, 5))
        self.assertEqual(x_test.shape, (5000, 7))
        self.assertEqual(t_test.shape, (5000, 5))

    def test_get_vocab(self):
        (x_train, t_train), (x_test, t_test) = self.sequence.load_data(self.file_path)
        vocab = self.sequence.get_vocab()
        char_to_id, id_to_char = vocab
        self.assertEqual({
            '1': 0,
            '6': 1,
            '+': 2,
            '7': 3,
            '5': 4,
            ' ': 5,
            '_': 6,
            '9': 7,
            '2': 8,
            '0': 9,
            '3': 10,
            '8': 11,
            '4': 12
        }, char_to_id)
        self.assertEqual({
            0: '1',
            1: '6',
            2: '+',
            3: '7',
            4: '5',
            5: ' ',
            6: '_',
            7: '9',
            8: '2',
            9: '0',
            10: '3',
            11: '8',
            12: '4'
        }, id_to_char)
        self.assertEqual(''.join([id_to_char[c] for c in x_train[0]]), '71+118 ')
        self.assertEqual(''.join([id_to_char[c] for c in t_train[0]]), '_189 ')

if __name__ == '__main__':
    unittest.main()
