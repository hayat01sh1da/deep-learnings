from numpy.typing import NDArray
from typing import Any

class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        self.mask      = (x <= 0)
        out            = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: NDArray[Any]) -> NDArray[Any]:
        dout[self.mask] = 0
        dx              = dout
        return dx
