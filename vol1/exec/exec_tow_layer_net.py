import numpy as np
import sys
sys.path.append('./src')
sys.path.append('./src/lib')
from two_layer_net import TwoLayerNet
from gradient import numerical_gradient

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

x = np.random.randn(100, 784)
y = net.predict(x)
print(y)

x = np.random.randn(100, 784)
t = np.random.randn(100, 10)

grads = net.numerical_gradient(x, t)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
