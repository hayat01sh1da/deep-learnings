import numpy as np
from numpy.typing import NDArray
from typing import Any
from attention import Attention

class TimeAttention:
    def __init__(self) -> None:
        self.params            = []
        self.grads             = []
        self.layers            = []
        self.attention_weights = []

    def forward(self, hs_enc: NDArray[Any], hs_dec: NDArray[Any]) -> NDArray[Any]:
        N, T, H = hs_dec.shape
        out     = np.empty_like(hs_dec)
        for t in range(T):
            layer        = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        return out

    def backward(self, dout: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        for t in range(T):
            layer            = self.layers[t]
            dhs, dh          = layer.backward(dout[:, t, :])
            dhs_enc         += dhs
            dhs_dec[:, t, :] = dh
        return dhs_enc, dhs_dec
