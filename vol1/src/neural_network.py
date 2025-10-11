import sys
sys.path.append('./dataset')
import sys
sys.path.append('./dataset')
import numpy as np

try:
    import matplotlib.pylab as plt
except Exception:
    plt = None

import os
import numpy as np

try:
    import matplotlib.pylab as plt
except Exception:
    plt = None

try:
    from PIL import Image
except Exception:
    Image = None

import pickle


class NeuralNetwork:
    """Lightweight NeuralNetwork helper used by the tests.

    Provides pure helper methods (sigmoid, softmax, step_func, relu, matrix_product)
    and a few dataset-dependent helpers that fail-gracefully when dataset/weights
    are unavailable.
    """

    def __init__(self):
        pass

    def _save_image(self, x, y, func_name):
        if plt is None:
            return
        plt.figure()
        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'img', f'{func_name}.png'))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(self, a):
        a = np.array(a)
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a

    def _process_image(self, x_train, t_train):
        img = x_train[0]
        label = t_train[0]
        reshaped_img = img.reshape(28, 28)
        return img, label, reshaped_img

    def _show_image(self, img):
        if Image is None:
            return
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()

    def _get_test_data(self):
        # Defer importing dataset code until runtime; raise clear error if absent.
        try:
            from mnist import load_mnist
        except Exception as exc:
            raise RuntimeError('mnist dataset loader not available') from exc
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        return x_test, t_test

    def _init_network(self):
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'sample_weight.pkl')
        if not os.path.exists(weights_path):
            raise RuntimeError('sample_weight.pkl not found')
        with open(weights_path, 'rb') as f:
            network = pickle.load(f)
        return network

    def _predict(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = self._sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self._sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self._softmax(a3)
        return y

    def step_func(self, x):
        return np.array(x > 0, dtype=int)

    def relu(self, x):
        return np.maximum(0, x)

    def matrix_product(self, a, b):
        return np.dot(a, b)

    def evaluate(self):
        # Attempt to run evaluation; if dataset or weights are missing, return a clear fallback.
        try:
            x, t = self._get_test_data()
            network = self._init_network()
        except RuntimeError:
            return '0.00%'

        batch_size = 100
        accuracy_count = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = self._predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_count += np.sum(p == t[i:i+batch_size])
        return f'{(float(accuracy_count) / len(x) * 100):.2f}%'
