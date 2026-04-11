import numpy as np
from numpy.typing import NDArray
from typing import Any
from layers.rnn import RNN

class TimeRNN:
    def __init__(self, Wx: NDArray[Any], Wh: NDArray[Any], b: NDArray[Any], stateful: bool = False) -> None:
        self.params   = [Wx, Wh, b]
        self.grads    = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers   = None
        self.h        = None
        self.dh       = None
        self.stateful = stateful

    def set_state(self, h: NDArray[Any]) -> None:
        self.h = h

    def reset_state(self) -> None:
        self.h = None

    def forward(self, xs: NDArray[Any]) -> NDArray[Any]:
        Wx, Wh, b = self.params
        N, T, D   = xs.shape
        D, H      = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        for t in range(T):
            layer       = RNN(*self.params)
            self.h      = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs: NDArray[Any]) -> NDArray[Any]:
        Wx, Wh, b = self.params
        N, T, H   = dhs.shape
        D, H      = Wx.shape
        dxs       = np.empty((N, T, D), dtype='f')
        dh        = 0
        grads     = [0, 0, 0]
        for t in reversed(range(T)):
            layer        = self.layers[t]
            dx, dh       = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs
