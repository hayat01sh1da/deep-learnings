import numpy as np
from numpy.typing import NDArray
from typing import Any

class TimeDropout:
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        self.params        = []
        self.grads         = []
        self.dropout_ratio = dropout_ratio
        self.mask          = None
        self.train_flag    = True

    def forward(self, xs: NDArray[Any]) -> NDArray[Any]:
        if self.train_flag:
            flag      = np.random.rand(*xs.shape) > self.dropout_ratio
            scale     = 1 / (1.0 - self.dropout_ratio)
            self.mask = flag.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        return dout * self.mask
