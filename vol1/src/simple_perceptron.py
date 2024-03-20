import numpy as np

class SimplePerceptron:
    def __init__(self):
        pass

    def and_gate(self, x1, x2):
        x = np.array([x1, x2])
        W = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(x * W) + b
        if tmp > 0:
            return 1
        else:
            return 0

    def nand_gate(self, x1, x2):
        x = np.array([x1, x2])
        W = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(x * W) + b
        if tmp > 0:
            return 1
        else:
            return 0

    def or_gate(self, x1, x2):
        x = np.array([x1, x2])
        W = np.array([0.8, 0.8])
        b = -0.7
        tmp = np.sum(x * W) + b
        if tmp > 0:
            return 1
        else:
            return 0
