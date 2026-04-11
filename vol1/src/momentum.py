import numpy as np
from numpy.typing import NDArray
from typing import Any

class Momentum:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.lr       = learning_rate
        self.momentum = momentum
        self.v        = None

    def update(self, params: dict[str, NDArray[Any]], grads: dict[str, NDArray[Any]]) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
