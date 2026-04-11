import numpy as np
from numpy.typing import NDArray
from typing import Any

class Affine:
    def __init__(self, W: NDArray[Any], b: NDArray[Any]) -> None:
        self.params = [W, b]
        self.grads  = [np.zeros_like(W), np.zeros_like(b)]
        self.x      = None

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        W, b   = self.params
        out    =  np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        W, b               = self.params
        dx                 = np.dot(dout, W.T)
        dW                 = np.dot(self.x.T, dout)
        db                 = np.sum(dout, axis = 0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
