class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
