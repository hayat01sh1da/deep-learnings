import sys
sys.path.append('./dataset')
import numpy as np
import matplotlib.pylab as plt
from mnist import load_mnist
from PIL import Image
import pickle

class NeuralNetwork:
    def __init__(self):
        pass

    def _save_image(self, x, y, func_name):
        plt.figure()
        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig('../img/{}.png'.format(func_name))

    def _sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def _softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def _process_image(self, x_train, t_train):
        img   = x_train[0]
        label = t_train[0]
        reshaped_img = img.reshape(28, 28)
        return img, label, reshaped_img

    def _show_image(self, img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()

    def _get_test_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        return x_test, t_test

    def _init_network(self):
        with open('./dataset/sample_weight.pkl', 'rb')as f:
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
        y  = self._softmax(a3)
        return y

    def step_func(self, x):
        y =  np.array(x > 0, dtype=int)
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def relu(self, x):
        y = np.maximum(0, x)
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def matrix_product(self, a, b):
        return np.dot(a, b)

    def evaluate(self):
        x, t = self._get_test_data()
        network = self._init_network()
        batch_size = 100
        accuracy_count = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = self._predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_count += np.sum(p == t[i:i+batch_size])
        return '{}%'.format(((float(accuracy_count) / len(x)) * 100))
