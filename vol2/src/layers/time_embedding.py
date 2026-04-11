import numpy as np
from numpy.typing import NDArray
from typing import Any
from layers.embedding import Embedding

class TimeEmbedding:
    def __init__(self, W: NDArray[Any]) -> None:
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.layers = None
        self.W      = W

    def forward(self, xs: NDArray[Any]) -> NDArray[Any]:
        N, T        = xs.shape
        V, D        = self.W.shape
        out         = np.empty((N, T, D), dtype='f')
        self.layers = []
        for t in range(T):
            layer        = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backward(self, dout: NDArray[Any]) -> None:
        N, T, D = dout.shape
        grad    = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad          += layer.grads[0]
        self.grads[0][...] = grad
        return None
