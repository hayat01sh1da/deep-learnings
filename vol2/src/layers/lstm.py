import numpy as np
import sys
from sigmoid import Sigmoid

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params  = [Wx, Wh, b]
        self.grads   = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache   = None
        self.sigmoid = Sigmoid()

    def _slice(self, A, H):
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        return f, g, i, o

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b  = self.params
        N, H       = h_prev.shape
        A          = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        f, g, i, o = self._slice(A, H)
        f          = self.sigmoid.forward(f)
        g          = np.tanh(g)
        i          = self.sigmoid.forward(i)
        o          = self.sigmoid.forward(o)
        c_next     = f * c_prev + g * i
        h_next     = o * np.tanh(c_next)
        self.cache = (x, h_prev, c_prev, (i, f, g, o), c_next)
        return c_next, h_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b                        = self.params
        x, h_prev, c_prev, gates, c_next = self.cache
        i, f, g, o                       = gates
        tanh_c_next                      = np.tanh(c_next)
        ds                               = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        dc_prev                          = ds * f
        di                               = ds * g
        df                               = ds * c_prev
        do                               = dh_next * tanh_c_next
        dg                               = ds * i
        di                               *= i * (1 - i)
        df                               *= f * (1 - f)
        do                               *= o * (1 - o)
        dg                               *= (1 - g ** 2)
        dA                               = np.hstack((df, dg, di, do))
        dWh                              = np.dot(h_prev.T, dA)
        dWx                              = np.dot(x.T, dA)
        db                               = dA.sum(axis = 0)
        self.grads[0][...]               = dWx
        self.grads[1][...]               = dWh
        self.grads[2][...]               = db
        dx                               = np.dot(dA, Wx.T)
        dh_prev                          = np.dot(dA, Wh.T)
        return dx, dh_prev, dc_prev
