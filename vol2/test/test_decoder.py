import numpy as np
import pytest

from decoder import Decoder


@pytest.fixture
def decoder_setup():
    decoder = Decoder(13, 100, 100)
    xs = np.random.randint(0, 13, (13, 100))
    h = np.random.randn(13, 100)
    return decoder, xs, h


def test_forward(decoder_setup):
    decoder, xs, h = decoder_setup
    assert decoder.forward(xs, h).shape == (13, 100, 13)


def test_backward(decoder_setup):
    decoder, xs, h = decoder_setup
    dscore = decoder.forward(xs, h)
    assert decoder.backward(dscore).shape == (13, 100)


def test_generate(decoder_setup):
    decoder, _, _ = decoder_setup
    assert len(decoder.generate(np.random.randn(1, 100), 0, 10)) == 10
