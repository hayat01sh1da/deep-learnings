import numpy as np
import pytest

from encoder import Encoder


@pytest.fixture
def setup():
    return Encoder(13, 100, 100), np.random.randint(0, 13, (7, 3))


def test_forward(setup):
    encoder, xs = setup
    outputs = encoder.forward(xs)
    assert len(outputs) == 7
    for output in outputs:
        assert output.shape == (100,)


def test_backward(setup):
    encoder, xs = setup
    dh = encoder.forward(xs)
    assert encoder.backward(dh) is None
