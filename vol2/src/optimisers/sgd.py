from numpy.typing import NDArray
from typing import Any


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, params: list[NDArray[Any]], grads: list[NDArray[Any]]) -> None:
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
