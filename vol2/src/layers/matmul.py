import numpy as np
from numpy.typing import NDArray
from typing import Any

class MatMul:
    def __init__(self, W: NDArray[Any]) -> None:
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.x      = None

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        W,     = self.params
        out    = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        W,                 = self.params
        dx                 = np.dot(dout, W.T)
        dW                 = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
