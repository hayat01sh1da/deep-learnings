import pytest

from sequence import Sequence


FILE_PATH = '../texts/addition.txt'


@pytest.fixture
def sequence():
    return Sequence()


def test_text_to_dict(sequence):
    questions, answers = sequence._text_to_dict(FILE_PATH, [], [])
    assert len(questions) == 50000
    assert len(answers) == 50000


def test_update_vocab(sequence):
    sequence._update_vocab('16+75  _91')
    assert sequence.char_to_id == {
        '1': 0, '6': 1, '+': 2, '7': 3, '5': 4, ' ': 5, '_': 6, '9': 7,
    }
    assert sequence.id_to_char == {
        0: '1', 1: '6', 2: '+', 3: '7', 4: '5', 5: ' ', 6: '_', 7: '9',
    }


def test_create_numpy_array(sequence):
    questions, answers = sequence._text_to_dict(FILE_PATH, [], [])
    sequence._create_vocab_dict(questions, answers)
    x, t = sequence._create_numpy_array(questions, answers)
    assert x.shape == (50000, 7)
    assert t.shape == (50000, 5)


def test_load_data(sequence):
    (x_train, t_train), (x_test, t_test) = sequence.load_data(FILE_PATH)
    assert x_train.shape == (45000, 7)
    assert t_train.shape == (45000, 5)
    assert x_test.shape == (5000, 7)
    assert t_test.shape == (5000, 5)


def test_get_vocab(sequence):
    (x_train, t_train), _ = sequence.load_data(FILE_PATH)
    char_to_id, id_to_char = sequence.get_vocab()
    assert char_to_id == {
        '1': 0, '6': 1, '+': 2, '7': 3, '5': 4, ' ': 5, '_': 6, '9': 7,
        '2': 8, '0': 9, '3': 10, '8': 11, '4': 12,
    }
    assert id_to_char == {
        0: '1', 1: '6', 2: '+', 3: '7', 4: '5', 5: ' ', 6: '_', 7: '9',
        8: '2', 9: '0', 10: '3', 11: '8', 12: '4',
    }
    assert ''.join(id_to_char[c] for c in x_train[0]) == '71+118 '
    assert ''.join(id_to_char[c] for c in t_train[0]) == '_189 '
