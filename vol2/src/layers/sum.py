import numpy as np
from numpy.typing import NDArray
from typing import Any

class Sum:
    def __init__(self, D: int, N: int) -> None:
        self.D = D
        self.N = N

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.sum(x, axis = 0, keepdims = True)

    def backward(self, dy: NDArray[Any]) -> NDArray[Any]:
        return np.repeat(dy, self.N, axis = 0)
