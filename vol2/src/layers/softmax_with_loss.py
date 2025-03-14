import numpy as np
import sys
sys.path.append('../concerns')
from softmax import Softmax
from cross_entropy_error import *

class SoftMaxWithLoss:
    def __init__(self):
        self.params  = []
        self.grads   = []
        self.y       = None
        self.t       = None
        self.softmax = Softmax()

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax.calc_softmax(x)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis = 1)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout = 1):
        batch_size                         = self.t.shape[0]
        dx                                 = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx                                *= dout
        dx                                 = dx / batch_size
        return dx
