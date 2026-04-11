import numpy as np
from numpy.typing import NDArray
from typing import Any

class Dropout:
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask          = None

    def forward(self, x: NDArray[Any], train_flag: bool = True) -> NDArray[Any]:
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        return dout * self.mask
