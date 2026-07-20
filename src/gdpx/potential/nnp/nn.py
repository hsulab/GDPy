import numpy as np


class SimpleNN:

    def __init__(self, n_input, n_hidden1=64, n_hidden2=64, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scale1 = np.sqrt(2.0 / n_input)
        self.W1 = np.random.randn(n_input, n_hidden1) * scale1
        self.b1 = np.zeros(n_hidden1)
        scale2 = np.sqrt(2.0 / n_hidden1)
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * scale2
        self.b2 = np.zeros(n_hidden2)
        scale3 = np.sqrt(2.0 / n_hidden2)
        self.W3 = np.random.randn(n_hidden2, 1) * scale3
        self.b3 = np.zeros(1)
        self._cache = None

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        y = h2 @ self.W3 + self.b3
        self._cache = (x, z1, h1, z2, h2, y)
        return y[:, 0]

    def gradient(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        W3_col = self.W3[:, 0]
        d2 = W3_col[None, :] * (1.0 - h2 ** 2)
        d1 = (d2 @ self.W2.T) * (1.0 - h1 ** 2)
        return d1 @ self.W1.T

    def energy_and_gradient(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        y = h2 @ self.W3 + self.b3
        W3_col = self.W3[:, 0]
        d2 = W3_col[None, :] * (1.0 - h2 ** 2)
        d1 = (d2 @ self.W2.T) * (1.0 - h1 ** 2)
        dy_dx = d1 @ self.W1.T
        self._cache = (x, z1, h1, z2, h2, y)
        return y[:, 0], dy_dx

    def backward(self, grad_output=None):
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        x, z1, h1, z2, h2, y = self._cache
        if grad_output is None:
            grad_output = np.ones(y.shape[0])
        dy = grad_output[:, None]

        dW3 = h2.T @ dy
        db3 = np.sum(dy, axis=0)

        dh2 = dy @ self.W3.T
        dz2 = dh2 * (1.0 - h2 ** 2)

        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (1.0 - h1 ** 2)

        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def update(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        self.W3 -= lr * grads["W3"]
        self.b3 -= lr * grads["b3"]

    def get_params(self):
        return dict(W1=self.W1.copy(), b1=self.b1.copy(),
                    W2=self.W2.copy(), b2=self.b2.copy(),
                    W3=self.W3.copy(), b3=self.b3.copy())

    def set_params(self, params):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        self.W3 = params["W3"].copy()
        self.b3 = params["b3"].copy()
