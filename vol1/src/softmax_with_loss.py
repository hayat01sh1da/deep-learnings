import numpy as np
from numpy.typing import NDArray
from typing import Any
import sys
sys.path.append('./lib')
from functions import softmax, cross_entropy_error

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y    = None
        self.t    = None

    def forward(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        self.t    = t
        self.y    = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout: int = 1) -> NDArray[Any]:
        batch_size = self.t.shape[0]
        dx         = (self.y - self.t) / batch_size
        return dx
