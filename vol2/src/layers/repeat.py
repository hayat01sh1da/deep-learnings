import numpy as np
from numpy.typing import NDArray
from typing import Any

class Repeat:
    def __init__(self, D: int, N: int) -> None:
        self.D = D
        self.N = N

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.repeat(x, self.N, axis = 0)

    def backward(self, dy: NDArray[Any]) -> NDArray[Any]:
        return np.sum(dy, axis = 0, keepdims = True)
