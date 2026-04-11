import numpy as np
from numpy.typing import NDArray
from typing import Any

class Matrix:
    def __init__(self, W: NDArray[Any]) -> None:
        self.W = W

    def calc_sum(self, X: NDArray[Any]) -> NDArray[Any]:
        sum = self.W + X
        return sum

    def calc_product(self, X: NDArray[Any]) -> NDArray[Any]:
        product = self.W * X
        return product

    def calc_scala_broadcast(self, num: float) -> NDArray[Any]:
        scala_broadcast = self.W * num
        return scala_broadcast

    def calc_array_broadcast(self, X: NDArray[Any]) -> NDArray[Any]:
        array_broadcast = self.W * X
        return array_broadcast

    def calc_inner_product(self, X: NDArray[Any]) -> NDArray[Any]:
        inner_product = np.dot(self.W, X)
        return inner_product
