import numpy as np
import sys
from numpy.typing import NDArray
from typing import Any
sys.path.append('../concerns')
from softmax import Softmax
from cross_entropy_error import *

class SoftMaxWithLoss:
    def __init__(self) -> None:
        self.params  = []
        self.grads   = []
        self.y       = None
        self.t       = None
        self.softmax = Softmax()

    def forward(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        self.t = t
        self.y = self.softmax.calc_softmax(x)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis = 1)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout: int = 1) -> NDArray[Any]:
        batch_size                         = self.t.shape[0]
        dx                                 = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx                                *= dout
        dx                                 = dx / batch_size
        return dx
