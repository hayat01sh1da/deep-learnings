import numpy as np
from numpy.typing import NDArray
from typing import Any
import sys
sys.path.append('./lib')
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient

class SimpleNet:
    def __init__(self) -> None:
        # Initialise with Gaussian distribution
        self.W = np.random.randn(2, 3)

    def predict(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.dot(x, self.W)

    def loss(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        z    = self.predict(x)
        y    = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
