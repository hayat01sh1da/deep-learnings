import numpy as np
from numpy.typing import NDArray
from typing import Any

class Vector:
    def __init__(self, x: NDArray[Any]) -> None:
        self.x = x

    def calc_inner_product(self, y: NDArray[Any]) -> NDArray[Any]:
        inner_product = np.dot(self.x, y)
        return inner_product
