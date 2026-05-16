import numpy as np
import pytest

from attention_encoder import AttentionEncoder


@pytest.fixture
def encoder_setup():
    encoder = AttentionEncoder(13, 100, 100)
    xs = np.random.randint(0, 13, (7, 3))
    return encoder, xs


def test_forward(encoder_setup):
    encoder, xs = encoder_setup
    assert encoder.forward(xs).shape == (7, 3, 100)


def test_backward(encoder_setup):
    encoder, xs = encoder_setup
    dhs = encoder.forward(xs)
    assert encoder.backward(dhs) is None
