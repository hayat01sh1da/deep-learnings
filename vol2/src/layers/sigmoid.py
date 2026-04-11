import numpy as np
from numpy.typing import NDArray
from typing import Any

class Sigmoid:
    def __init__(self) -> None:
        self.params = []
        self.grads  = []
        self.out    = None

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        out      = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        dx = dout * (1.0 - self.out) * self.out
        return dx
