import numpy as np


class SimpleNN:

    def __init__(self, n_input, hidden_sizes=(64, 64), seed=None):
        if seed is not None:
            np.random.seed(seed)
        sizes = [n_input] + list(hidden_sizes) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.biases.append(np.zeros(sizes[i + 1]))
        self._cache_acts = None
        self._cache_y = None

    def forward(self, x):
        acts = [x]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        y = acts[-1] @ self.weights[-1] + self.biases[-1]
        self._cache_acts = acts
        self._cache_y = y
        return y[:, 0]

    def gradient(self, x):
        acts = [x]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        n = len(self.weights)
        g = np.ones((x.shape[0], 1)) @ self.weights[-1].T
        for i in range(n - 2, -1, -1):
            g = g * (1.0 - acts[i + 1] ** 2)
            g = g @ self.weights[i].T
        return g

    def energy_and_gradient(self, x):
        acts = [x]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        y = acts[-1] @ self.weights[-1] + self.biases[-1]
        self._cache_acts = acts
        self._cache_y = y

        n = len(self.weights)
        g = np.ones((x.shape[0], 1)) @ self.weights[-1].T
        for i in range(n - 2, -1, -1):
            g = g * (1.0 - acts[i + 1] ** 2)
            g = g @ self.weights[i].T
        return y[:, 0], g

    def backward(self, grad_output=None):
        if self._cache_acts is None:
            raise RuntimeError("forward must be called before backward")
        acts = self._cache_acts
        y = self._cache_y
        if grad_output is None:
            grad_output = np.ones(y.shape[0])
        g = grad_output[:, None]
        n = len(self.weights)

        d_weights = [None] * n
        d_biases = [None] * n

        h_prev = acts[-1]
        d_weights[-1] = h_prev.T @ g
        d_biases[-1] = np.sum(g, axis=0)
        g = g @ self.weights[-1].T

        for i in range(n - 2, -1, -1):
            g = g * (1.0 - acts[i + 1] ** 2)
            h_prev = acts[i]
            d_weights[i] = h_prev.T @ g
            d_biases[i] = np.sum(g, axis=0)
            if i > 0:
                g = g @ self.weights[i].T

        return {"weights": d_weights, "biases": d_biases}

    def update(self, grads, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads["weights"][i]
            self.biases[i] -= lr * grads["biases"][i]

    def get_params(self):
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params):
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]
