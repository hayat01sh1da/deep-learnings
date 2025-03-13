import numpy as np
import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

class NeuralNetworkLearning:
    def __init__(self):
        pass

    def _get_train_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
        return x_train, t_train

    def _random_choice(self):
        x_train, t_train = self._get_train_data()
        train_size       = x_train.shape[0]
        batch_size       = 10
        batch_mask       = np.random.choice(train_size, batch_size)
        x_batch          = x_train[batch_mask]
        t_batch          = t_train[batch_mask]
        return x_batch, t_batch

    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        delta      = 1e-7
        # One-hot
        return -np.sum(t * np.log(y + delta)) / batch_size
        # Non One-hot
        # return -np.sum(t * np.log(y[np.arrage(batch_size), t] + delta)) / batch_size

    def numerical_diff(self, f, x):
        h = 1e-4
        return (f(x + h) - f(x - h)) / (2 * h)

    def numerial_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)
        for index in range(x.size):
            tmp_val = x[index]
            # Calculate f(x + h)
            x[index] = tmp_val + h
            fxh1 = f(x)
            # Calculate f(x - h)
            x[index] = tmp_val - h
            fxh2 = f(x)
            grad[index] = (fxh1 - fxh2) / (2 * h)
            # Revert value
            x[index] = tmp_val
        return grad

    def gradient_descent(self, f, init_x, learning_rate=0.01, step_num=100):
        x = init_x
        for i in range(step_num):
            grad = self.numerial_gradient(f, x)
            x   -= learning_rate * grad
        return x
