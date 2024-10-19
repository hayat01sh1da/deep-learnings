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

train_loss_list = []
train_acc_list  = []
test_acc_list   = []

# Hyper params
iters_num      = 10000
train_size     = x_train.shape[0]
batch_size     = 100
learning_rate  = 0.1
iter_per_epoch = max(train_size / batch_size, 1)

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # Get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch    = x_train[batch_mask]
    t_batch    = t_train[batch_mask]

    # Calculate gradient
    grad = net.gradient(x_batch, t_batch)

    # Update params
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    # Record learning process
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc  = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | {}, {}'.format(str(train_acc), str(test_acc)))

plt.figure()
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig('../img/{}.png'.format('train_neural_network'))

pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)
for pycache in pycaches:
    if os.path.exists(pycache):
        shutil.rmtree(pycache)
