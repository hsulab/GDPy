import numpy as np


class SimpleNN:

    def __init__(self, n_input, hidden_sizes=(64, 64), seed=None, rng=None):
        if rng is not None and seed is not None:
            raise ValueError("Pass either seed or rng, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        sizes = [n_input] + list(hidden_sizes) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            # Xavier scaling keeps tanh units out of saturation when inputs
            # have unit variance.
            scale = np.sqrt(1.0 / sizes[i])
            self.weights.append(rng.normal(size=(sizes[i], sizes[i + 1])) * scale)
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

    def adam_update(self, grads, lr, state, beta1=0.9, beta2=0.999, eps=1e-8):
        """Apply one Adam step (updates ``state`` in place).

        Args:
            grads: Dict with keys ``weights`` and ``biases``.
            lr: Learning rate.
            state: Dict with keys ``m_w``, ``v_w``, ``m_b``, ``v_b``, ``t``.
                Created lazily if ``None``.

        Returns:
            The (possibly newly created) Adam state.
        """
        if state is None:
            state = {
                "m_w": [np.zeros_like(w) for w in self.weights],
                "v_w": [np.zeros_like(w) for w in self.weights],
                "m_b": [np.zeros_like(b) for b in self.biases],
                "v_b": [np.zeros_like(b) for b in self.biases],
                "t": 0,
            }
        t = state["t"] + 1
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        for i in range(len(self.weights)):
            gw = grads["weights"][i]
            mw = state["m_w"][i]
            vw = state["v_w"][i]
            mw[...] = beta1 * mw + (1.0 - beta1) * gw
            vw[...] = beta2 * vw + (1.0 - beta2) * gw**2
            self.weights[i] -= lr * (mw / bc1) / (np.sqrt(vw / bc2) + eps)

            gb = grads["biases"][i]
            mb = state["m_b"][i]
            vb = state["v_b"][i]
            mb[...] = beta1 * mb + (1.0 - beta1) * gb
            vb[...] = beta2 * vb + (1.0 - beta2) * gb**2
            self.biases[i] -= lr * (mb / bc1) / (np.sqrt(vb / bc2) + eps)
        state["t"] = t
        return state

    def get_params(self):
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params):
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]


class ElementwiseNN:
    """One normalized atomic-energy network per central element."""

    def __init__(
        self,
        n_input,
        elements,
        hidden_sizes=(64, 64),
        feature_mean=None,
        feature_scale=None,
        atomic_offsets=None,
        rng=None,
    ):
        self.elements = [str(element) for element in elements]
        if not self.elements:
            raise ValueError("ElementwiseNN requires at least one element.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError(f"Duplicate elements are not allowed: {self.elements}.")
        self.element_index = {element: i for i, element in enumerate(self.elements)}
        self.n_input = int(n_input)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if rng is None:
            rng = np.random.default_rng()

        shape = (len(self.elements), self.n_input)
        self.feature_mean = (
            np.zeros(shape, dtype=float)
            if feature_mean is None
            else np.asarray(feature_mean, dtype=float).copy()
        )
        self.feature_scale = (
            np.ones(shape, dtype=float)
            if feature_scale is None
            else np.asarray(feature_scale, dtype=float).copy()
        )
        self.atomic_offsets = (
            np.zeros(len(self.elements), dtype=float)
            if atomic_offsets is None
            else np.asarray(atomic_offsets, dtype=float).copy()
        )
        if self.feature_mean.shape != shape or self.feature_scale.shape != shape:
            raise ValueError(
                "feature_mean and feature_scale must have shape "
                f"{shape}; got {self.feature_mean.shape} and "
                f"{self.feature_scale.shape}."
            )
        if self.atomic_offsets.shape != (len(self.elements),):
            raise ValueError(
                f"atomic_offsets must have shape {(len(self.elements),)}; "
                f"got {self.atomic_offsets.shape}."
            )
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("All descriptor scales must be positive.")

        self.networks = {
            element: SimpleNN(self.n_input, self.hidden_sizes, rng=rng)
            for element in self.elements
        }
        self._last_indices = None

    def _indices(self, symbols):
        symbols = np.asarray(symbols, dtype=str)
        unknown = sorted(set(symbols) - set(self.elements))
        if unknown:
            raise ValueError(
                f"Unknown central elements {unknown}; model elements are "
                f"{self.elements}."
            )
        return symbols, {
            element: np.flatnonzero(symbols == element)
            for element in self.elements
        }

    def _normalized(self, descriptors, element, indices):
        element_index = self.element_index[element]
        return (
            descriptors[indices] - self.feature_mean[element_index]
        ) / self.feature_scale[element_index]

    def forward(self, descriptors, symbols, include_offsets=True):
        descriptors = np.asarray(descriptors, dtype=float)
        symbols, indices = self._indices(symbols)
        if descriptors.shape != (len(symbols), self.n_input):
            raise ValueError(
                f"descriptors have shape {descriptors.shape}; expected "
                f"{(len(symbols), self.n_input)}."
            )
        energies = np.zeros(len(symbols), dtype=float)
        for element, atom_indices in indices.items():
            element_index = self.element_index[element]
            values = self.networks[element].forward(
                self._normalized(descriptors, element, atom_indices)
            )
            if include_offsets:
                values = values + self.atomic_offsets[element_index]
            energies[atom_indices] = values
        self._last_indices = indices
        return energies

    def energy_and_gradient(self, descriptors, symbols, include_offsets=True):
        descriptors = np.asarray(descriptors, dtype=float)
        symbols, indices = self._indices(symbols)
        if descriptors.shape != (len(symbols), self.n_input):
            raise ValueError(
                f"descriptors have shape {descriptors.shape}; expected "
                f"{(len(symbols), self.n_input)}."
            )
        energies = np.zeros(len(symbols), dtype=float)
        gradient = np.zeros_like(descriptors)
        for element, atom_indices in indices.items():
            element_index = self.element_index[element]
            values, normalized_gradient = self.networks[element].energy_and_gradient(
                self._normalized(descriptors, element, atom_indices)
            )
            if include_offsets:
                values = values + self.atomic_offsets[element_index]
            energies[atom_indices] = values
            gradient[atom_indices] = (
                normalized_gradient / self.feature_scale[element_index]
            )
        self._last_indices = indices
        return energies, gradient

    def backward(self, grad_output):
        if self._last_indices is None:
            raise RuntimeError("forward must be called before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        return {
            element: self.networks[element].backward(grad_output[atom_indices])
            for element, atom_indices in self._last_indices.items()
        }

    def double_backward(self, descriptor_seed):
        if self._last_indices is None:
            raise RuntimeError(
                "energy_and_gradient must be called before double_backward"
            )
        descriptor_seed = np.asarray(descriptor_seed, dtype=float)
        gradients = {}
        for element, atom_indices in self._last_indices.items():
            element_index = self.element_index[element]
            gradients[element] = self.networks[element].double_backward(
                descriptor_seed[atom_indices]
                / self.feature_scale[element_index]
            )
        return gradients

    def adam_update(self, gradients, learning_rate, states):
        if states is None:
            states = {element: None for element in self.elements}
        for element in self.elements:
            states[element] = self.networks[element].adam_update(
                gradients[element], learning_rate, states[element]
            )
        return states

    def get_params(self):
        return {
            element: self.networks[element].get_params()
            for element in self.elements
        }

    def set_params(self, params):
        for element in self.elements:
            self.networks[element].set_params(params[element])

    def save_parameters(self, destination):
        for element_index, element in enumerate(self.elements):
            params = self.networks[element].get_params()
            for layer, weight in enumerate(params["weights"]):
                destination[f"W_{element_index}_{layer}"] = weight
            for layer, bias in enumerate(params["biases"]):
                destination[f"b_{element_index}_{layer}"] = bias

    def load_parameters(self, source):
        for element_index, element in enumerate(self.elements):
            params = {
                "weights": [
                    source[f"W_{element_index}_{layer}"]
                    for layer in range(len(self.hidden_sizes) + 1)
                ],
                "biases": [
                    source[f"b_{element_index}_{layer}"]
                    for layer in range(len(self.hidden_sizes) + 1)
                ],
            }
            self.networks[element].set_params(params)
