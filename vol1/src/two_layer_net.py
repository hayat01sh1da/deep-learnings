import numpy as np
from numpy.typing import NDArray
from typing import Any
import sys
sys.path.append('./lib')
from layers import *
from gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_unit_std: float = 0.01) -> None:
        self.params            = {}
        self.params['W1']      = weight_unit_std * np.random.randn(input_size, hidden_size)
        self.params['b1']      = np.zeros(hidden_size)
        self.params['W2']      = weight_unit_std * np.random.randn(hidden_size, output_size)
        self.params['b2']      = np.zeros(output_size)
        self.layers            = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu']    = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer        = SoftmaxWithLoss()

    def predict(self, x: NDArray[Any]) -> NDArray[Any]:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x: NDArray[Any], t: NDArray[Any]) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: NDArray[Any], t: NDArray[Any]) -> dict[str, NDArray[Any]]:
        loss_W      = lambda W: self.loss(x, t)
        grads       = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x: NDArray[Any], t: NDArray[Any]) -> dict[str, NDArray[Any]]:
        # Forward
        self.loss(x, t)
        # Backward
        dout   = 1
        dout   = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads       = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
