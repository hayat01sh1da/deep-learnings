import numpy as np

from affine import Affine


def test_forward():
    affine = Affine(np.random.randn(2, 4), np.random.randn(4))
    assert affine.forward(np.random.randn(4, 2)).shape == (4, 4)


def test_backward():
    affine = Affine(np.random.randn(2, 4), np.random.randn(4))
    affine.forward(np.random.randn(4, 2))
    assert affine.backward(np.random.randn(4, 4)).shape == (4, 2)
