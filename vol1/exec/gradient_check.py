import numpy as np
import matplotlib.pylab as plt
import sys
import os
import shutil
import glob
sys.path.append('./dataset')
sys.path.append('./src')
sys.path.append('./src/lib')
from mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = net.numerical_gradient(x_batch, t_batch)
grad_backprop  = net.gradient(x_batch, t_batch)

# Calculate absolute diff of each weight
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f'{key}: {str(diff)}')

pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)
for pycache in pycaches:
    if os.path.exists(pycache):
        shutil.rmtree(pycache)
