import numpy as np
from numpy.typing import NDArray
from typing import Any

class RNN:
    def __init__(self, Wx: NDArray[Any], Wh: NDArray[Any], b: NDArray[Any]) -> None:
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache  = None

    def forward(self, x: NDArray[Any], h_prev: NDArray[Any]) -> NDArray[Any]:
        Wx, Wh, b  = self.params
        t          = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next     = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        Wx, Wh, b          = self.params
        x, h_prev, h_next  = self.cache
        dt                 = dh_next * (1 - h_next ** 2)
        db                 = np.sum(dt, axis = 0)
        dWh                = np.dot(h_prev.T, dt)
        dh_prev            = np.dot(dt, Wh.T)
        dWx                = np.dot(x.T, dt)
        dx                 = np.dot(dt, Wx.T)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev
