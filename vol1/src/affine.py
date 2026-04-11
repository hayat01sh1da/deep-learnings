import numpy as np
from numpy.typing import NDArray
from typing import Any

class Affine:
    def __init__(self, W: NDArray[Any], b: NDArray[Any]) -> None:
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        self.x = x
        out    = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        dx      = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx
