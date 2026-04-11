import numpy as np
from numpy.typing import NDArray
from typing import Any

class AdaGrad:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.lr = learning_rate
        self.h  = None

    def update(self, params: dict[str, NDArray[Any]], grads: dict[str, NDArray[Any]]) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
