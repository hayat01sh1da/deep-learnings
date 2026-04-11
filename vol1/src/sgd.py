from numpy.typing import NDArray
from typing import Any

class SGD:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.lr: float = learning_rate

    def update(self, params: dict[str, NDArray[Any]], grads: dict[str, NDArray[Any]]) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]
