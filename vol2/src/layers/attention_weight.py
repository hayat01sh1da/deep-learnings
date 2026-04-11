import numpy as np
from numpy.typing import NDArray
from typing import Any
from softmax import Softmax

class AttentionWeight:
    def __init__(self) -> None:
        self.cache   = None
        self.softmax = Softmax()

    def forward(self, hs: NDArray[Any], h: NDArray[Any]) -> NDArray[Any]:
        N, T, H = hs.shape
        hr      = h.reshape(N, 1, H).repeat(T, axis = 1)
        t       = hs * hr
        s       = np.sum(t, axis = 2)
        self.softmax.forward(s)
        a          = self.softmax.out
        self.cache = (hs, hr)
        return a

    def backward(self, da: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        hs, hr  = self.cache
        N, T, H = hs.shape
        ds      = self.softmax.backward(da)
        dt      = ds.reshape(N, T, 1).repeat(H, axis = 2)
        dhs     = dt * hr
        dh      = np.sum(dhs, axis = 1)
        return dhs, da
