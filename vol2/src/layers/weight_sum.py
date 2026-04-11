import numpy as np
from numpy.typing import NDArray
from typing import Any

class WeightSum:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, hs: NDArray[Any], a: NDArray[Any]) -> NDArray[Any]:
        N, T, H    = hs.shape
        ar         = a.reshape(N, T, 1).repeat(H, axis = 2)
        t          = hs * ar
        c          = np.sum(t, axis = 1)
        self.cache = (hs, ar)
        return c

    def backward(self, dc: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        hs, ar  = self.cache
        N, T, H = hs.shape
        dt      = dc.reshape(N, 1, H).repeat(T, axis = 1)
        dar     = dt * hs
        dhs     = dt * ar
        da      = np.sum(dar, axis = 2)
        return dhs, da
