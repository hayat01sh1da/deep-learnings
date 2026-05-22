import os
import sys

from invoke import Context, task

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ('src', 'src/lib', 'dataset'):
    sys.path.insert(0, os.path.join(_ROOT, _sub))


@task
def simple_net(c: Context) -> None:
    """Run the SimpleNet gradient demo"""
    import numpy as np
    from gradient import numerical_gradient
    from simple_net import SimpleNet

    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    def f(w):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)


@task
def two_layer_net(c: Context) -> None:
    """Run the TwoLayerNet shape demo"""
    import numpy as np
    from two_layer_net import TwoLayerNet

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


@task
def gradient_check(c: Context) -> None:
    """Compare numerical and backprop gradients"""
    import numpy as np
    from mnist import load_mnist
    from two_layer_net import TwoLayerNet

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = net.numerical_gradient(x_batch, t_batch)
    grad_backprop = net.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f'{key}: {str(diff)}')


@task
def train_neural_network(c: Context) -> None:
    """Train a two-layer net on MNIST and plot accuracy"""
    import matplotlib.pylab as plt
    import numpy as np
    from mnist import load_mnist
    from two_layer_net import TwoLayerNet

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Hyper params
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    iter_per_epoch = max(train_size / batch_size, 1)

    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        # Get mini batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

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
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f'train acc, test acc | {str(train_acc)}, '
                  f'{str(test_acc)}')

    plt.figure()
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(f'../img/{"train_neural_network"}.png')


@task(default=True)
def test(c: Context) -> None:
    """Run all tests"""
    c.run('pytest .')
