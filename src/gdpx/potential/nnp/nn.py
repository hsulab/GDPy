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

    def double_backward(self, seed):
        """Second-order backward pass.

        Given the cached forward pass, compute the gradient of
            Q = sum_{j,k} seed[j,k] * g[j,k]
        with respect to all weights and biases, where ``g = dE/dx`` is the
        input-layer gradient produced by ``backward()`` (seed = ones over
        atoms).  This is the mixed second derivative ``d^2E/dx dW`` contracted
        with ``seed`` and is needed for analytic force training.

        Args:
            seed: Array of shape ``(n_atoms, n_features)``.

        Returns:
            Dict with keys ``weights`` and ``biases`` holding ``dQ/dW`` and
            ``dQ/db``.
        """
        if self._cache_acts is None:
            raise RuntimeError("forward must be called before double_backward")
        acts = self._cache_acts
        n = len(self.weights)
        m = seed.shape[0]

        # Values of the first-order backward g-vectors.
        # u[i] = dE/da_i  (derivative wrt activation of layer i)
        # v[i] = dE/dz_{i+1} = u[i+1] * (1 - acts[i+1]^2)
        u = [None] * n
        v = [None] * (n - 1)
        u[n - 1] = np.ones((m, 1)) @ self.weights[n - 1].T
        for i in range(n - 2, -1, -1):
            v[i] = u[i + 1] * (1.0 - acts[i + 1] ** 2)
            u[i] = v[i] @ self.weights[i].T

        dQ_dW = [np.zeros_like(w) for w in self.weights]
        dQ_db = [np.zeros_like(b) for b in self.biases]
        dQ_dacts = [np.zeros_like(a) for a in acts]

        # Reverse mode through the backward chain (u[i] = v[i] @ W[i].T).
        A = seed  # dQ/du[0]
        for i in range(n - 1):
            C = A @ self.weights[i]  # dQ/dv[i]
            dQ_dW[i] += A.T @ v[i]  # from u[i] = v[i] @ W[i].T
            # v[i] = u[i+1] * (1 - acts[i+1]**2)
            dQ_dacts[i + 1] += C * u[i + 1] * (-2.0 * acts[i + 1])
            A = C * (1.0 - acts[i + 1] ** 2)
        # u[n-1] = ones @ W[n-1].T
        dQ_dW[n - 1] += A.T @ np.ones((m, 1))

        # Propagate through the forward activations (acts[i+1] = tanh(...)).
        for i in range(n - 2, -1, -1):
            t = dQ_dacts[i + 1] * (1.0 - acts[i + 1] ** 2)
            dQ_dW[i] += acts[i].T @ t
            dQ_db[i] += t.sum(axis=0)
            if i > 0:
                dQ_dacts[i] += t @ self.weights[i].T

        return {"weights": dQ_dW, "biases": dQ_db}

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
