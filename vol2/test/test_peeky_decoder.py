import numpy as np
import pytest

from peeky_decoder import PeekyDecoder


@pytest.fixture
def setup():
    decoder = PeekyDecoder(13, 16, 128)
    return decoder, np.random.randint(0, 13, (13, 16)), np.random.randn(13, 128)


def test_forward(setup):
    decoder, xs, h = setup
    assert decoder.forward(xs, h).shape == (13, 16, 13)


def test_backward(setup):
    decoder, xs, h = setup
    dscore = decoder.forward(xs, h)
    assert decoder.backward(dscore).shape == (13, 128)


def test_generate(setup):
    decoder, _, _ = setup
    assert len(decoder.generate(np.random.randn(1, 128), 0, 10)) == 10
