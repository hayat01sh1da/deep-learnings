import numpy as np
from numpy.typing import NDArray
from typing import Any

class Adam:
    def __init__(self, learnig_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.lr    = learnig_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter  = 0
        self.m     = None
        self.v     = None

    def update(self, params: dict[str, NDArray[Any]], grads: dict[str, NDArray[Any]]) -> None:
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iter     += 1
        learnig_rate_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= learnig_rate_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
