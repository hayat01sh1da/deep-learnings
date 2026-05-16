import numpy as np

from matmul import MatMul


def test_forward():
    matmul = MatMul(np.random.rand(4, 2))
    assert matmul.forward(np.random.rand(2, 4)).shape == (2, 2)


def test_backward():
    matmul = MatMul(np.random.rand(4, 2))
    x = np.random.rand(2, 4)
    dout = matmul.forward(x)
    assert matmul.backward(dout).shape == (2, 4)
