import numpy as np
import sys
from numpy.typing import NDArray
from typing import Any
sys.path.append('./layers')
from sigmoid import Sigmoid

class NeuralNetwork:
    def __init__(self, x: NDArray[Any], W: NDArray[Any], b: NDArray[Any]) -> None:
        self.x = x
        self.W = W
        self.b = b

    def get_hidden_layer(self) -> NDArray[Any]:
        h = np.dot(self.x, self.W) + self.b
        return h

    def get_output_layer(self, h: NDArray[Any]) -> NDArray[Any]:
        sigmoid = Sigmoid()
        a       = sigmoid.forward(h)
        out     = np.dot(a, self.W) + self.b
        return out
