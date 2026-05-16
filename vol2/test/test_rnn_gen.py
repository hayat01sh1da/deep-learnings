import pytest

from ptb import load_data
from rnnlm_gen import RNNLMGen


@pytest.fixture
def setup():
    _, word_to_id, id_to_word = load_data('train')
    rnnlm_gen = RNNLMGen()
    rnnlm_gen.load_params('../pkl/better_rnnlm.pkl')
    start_id = word_to_id['you']
    skip_ids = [word_to_id[w] for w in ['N', '<unk>', '$']]
    return rnnlm_gen, id_to_word, start_id, skip_ids


def test_word_ids_list(setup):
    rnnlm_gen, _, start_id, skip_ids = setup
    assert len(rnnlm_gen.word_ids_list(start_id, skip_ids)) == 100


def test_generate_text(setup):
    rnnlm_gen, id_to_word, start_id, skip_ids = setup
    word_ids = rnnlm_gen.word_ids_list(start_id, skip_ids)
    text = rnnlm_gen.generate_text(id_to_word, word_ids)
    assert 750 < len(text) < 870
