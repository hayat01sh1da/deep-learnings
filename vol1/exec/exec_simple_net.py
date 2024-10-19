import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/lib')
from simple_net import SimpleNet
from gradient import numerical_gradient

net = SimpleNet()
print(net.W)
# >>> [[-0.44439281  0.30789016 -1.50579685]
#      [-0.93170709  0.08170439 -0.12740328]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# >>> [ 1.00824761 -1.47819523  0.03650346]

print(np.argmax(p))
# >> 1

t = np.array([0, 0, 1])
print(net.loss(x, t))
# >>> 1.704819611629646

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
# >>> [[ 0.09999078  0.39092591 -0.49091668]
#      [ 0.14998616  0.58638886 -0.73637502]]

pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)
for pycache in pycaches:
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
