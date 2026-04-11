import numpy as np
from numpy.typing import NDArray
from typing import Any

class Softmax:
    def __init__(self) -> None:
        self.out = None

    def calc_softmax(self, x: NDArray[Any]) -> NDArray[Any]:
        if x.ndim == 2:
            x  = x - x.max(axis = 1, keepdims = True)
            x  = np.exp(x)
            x /= x.sum(axis = 1, keepdims = True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x

    def forward(self, x: NDArray[Any]) -> None:
        self.out = self.calc_softmax(x)

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        dx    = self.out * dout
        sumdx = np.sum(dx, axis = 1, keepdims = True)
        dx   -= self.out * sumdx
        return dx
