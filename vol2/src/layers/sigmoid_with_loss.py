import numpy as np
import sys
from numpy.typing import NDArray
from typing import Any
sys.path.append('../concerns')
from cross_entropy_error import *

class SigmoidWithLoss:
    def __init__(self) -> None:
        self.params = []
        self.grads  = []
        self.y      = None
        self.t      = None

    def forward(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        loss   = cross_entropy_error(np.c_[1 - self.y, self.y], t)
        return loss

    def backward(self, dout: int = 1) -> NDArray[Any]:
        batch_size = self.t.shape[0]
        dx         = (self.y - self.t) * dout / batch_size
        return dx
