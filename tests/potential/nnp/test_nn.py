import numpy as np
from gdpx.potential.nnp.nn import ElementwiseNN, SimpleNN


class TestSimpleNN:
    def setup_method(self):
        np.random.seed(42)
        self.x = np.random.randn(5, 10)

    def _fd_gradient(self, nn, x, ai, fi, eps=1e-6):
        orig = x[ai, fi].copy()
        x[ai, fi] = orig + eps
        ep = nn.forward(x)[ai]
        x[ai, fi] = orig - eps
        em = nn.forward(x)[ai]
        x[ai, fi] = orig
        return (ep - em) / (2.0 * eps)

    def test_gradient_fd(self):
        for hs in [(), (8,), (8, 4), (16, 8, 4)]:
            nn = SimpleNN(10, hidden_sizes=hs)
            _, g = nn.energy_and_gradient(self.x)
            for ai in range(3):
                for fi in range(3):
                    fd = self._fd_gradient(nn, self.x, ai, fi)
                    assert abs(fd - g[ai, fi]) < 1e-5, \
                        f"hs={hs} ({ai},{fi}): FD={fd:.6f} ana={g[ai,fi]:.6f}"

    def test_forward_backward_roundtrip(self):
        for hs in [(), (16,), (16, 8), (32, 16, 8)]:
            nn = SimpleNN(10, hidden_sizes=hs)
            e1 = nn.forward(self.x)
            grads = nn.backward(np.ones(5))
            assert len(grads["weights"]) == len(hs) + 1
            assert len(grads["biases"]) == len(hs) + 1
            nn.update(grads, 0.01)
            e2 = nn.forward(self.x)
            assert not np.allclose(e1, e2)

    def test_get_set_params(self):
        nn1 = SimpleNN(10, hidden_sizes=(16, 8))
        nn1.forward(self.x)  # init cache
        p = nn1.get_params()
        assert len(p["weights"]) == 3
        assert len(p["biases"]) == 3

        nn2 = SimpleNN(10, hidden_sizes=(16, 8))
        nn2.set_params(p)
        e1 = np.sum(nn1.forward(self.x))
        e2 = np.sum(nn2.forward(self.x))
        assert abs(e1 - e2) < 1e-12

    def test_energy_and_gradient(self):
        nn = SimpleNN(10, hidden_sizes=(16, 8))
        e, g = nn.energy_and_gradient(self.x)
        e2 = nn.forward(self.x)
        assert np.allclose(e, e2)
        g2 = nn.gradient(self.x)
        assert np.allclose(g, g2)

    def test_double_backward_fd(self):
        eps = 1e-6
        for hs in [(), (8,), (8, 4), (16, 8, 4)]:
            nn = SimpleNN(6, hidden_sizes=hs, seed=42)
            x = np.random.randn(5, 6) * 1.2
            seed = np.random.randn(5, 6)

            def Q():
                nn.forward(x)
                return np.sum(seed * nn.gradient(x))

            nn.forward(x)
            dQ = nn.double_backward(seed)
            max_err = 0.0
            for wi, W in enumerate(nn.weights):
                for p in np.ndindex(W.shape):
                    old = W[p]
                    W[p] = old + eps
                    Qp = Q()
                    W[p] = old - eps
                    Qm = Q()
                    W[p] = old
                    max_err = max(max_err, abs((Qp - Qm) / (2.0 * eps) - dQ["weights"][wi][p]))
            for bi, B in enumerate(nn.biases):
                for p in np.ndindex(B.shape):
                    old = B[p]
                    B[p] = old + eps
                    Qp = Q()
                    B[p] = old - eps
                    Qm = Q()
                    B[p] = old
                    max_err = max(max_err, abs((Qp - Qm) / (2.0 * eps) - dQ["biases"][bi][p]))
            assert max_err < 1e-4, f"hs={hs}: max double-backward FD error = {max_err:.2e}"

    def test_adam_state_persists(self):
        nn = SimpleNN(3, hidden_sizes=(4,), seed=2)
        nn.forward(np.ones((2, 3)))
        gradients = nn.backward(np.ones(2))
        state = nn.adam_update(gradients, 0.01, None)
        state = nn.adam_update(gradients, 0.01, state)
        assert state["t"] == 2


class TestElementwiseNN:
    def test_central_elements_use_different_networks(self):
        model = ElementwiseNN(
            3,
            ["Cu", "Au"],
            hidden_sizes=(4,),
            feature_mean=np.zeros((2, 3)),
            feature_scale=np.ones((2, 3)),
            atomic_offsets=np.array([1.0, 2.0]),
            rng=np.random.default_rng(4),
        )
        descriptors = np.zeros((2, 3))
        energies = model.forward(descriptors, ["Cu", "Au"])
        assert energies[0] != energies[1]

    def test_normalized_gradient_matches_finite_difference(self):
        model = ElementwiseNN(
            3,
            ["Cu", "Au"],
            hidden_sizes=(4,),
            feature_mean=np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]),
            feature_scale=np.array([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]]),
            rng=np.random.default_rng(5),
        )
        descriptors = np.array([[1.2, 2.4, 3.8], [0.7, 1.4, 2.1]])
        symbols = ["Cu", "Au"]
        _, gradient = model.energy_and_gradient(descriptors, symbols)
        step = 1.0e-6
        for atom_index in range(2):
            for feature in range(3):
                original = descriptors[atom_index, feature]
                descriptors[atom_index, feature] = original + step
                plus = model.forward(descriptors, symbols).sum()
                descriptors[atom_index, feature] = original - step
                minus = model.forward(descriptors, symbols).sum()
                descriptors[atom_index, feature] = original
                assert abs(
                    gradient[atom_index, feature]
                    - (plus - minus) / (2.0 * step)
                ) < 1.0e-6

    def test_normalized_double_backward_matches_finite_difference(self):
        model = ElementwiseNN(
            2,
            ["Cu", "Au"],
            hidden_sizes=(3,),
            feature_mean=np.zeros((2, 2)),
            feature_scale=np.array([[2.0, 3.0], [1.5, 2.5]]),
            rng=np.random.default_rng(6),
        )
        descriptors = np.array([[0.2, 0.4], [0.6, 0.8]])
        symbols = ["Cu", "Au"]
        seed = np.array([[0.5, -0.2], [0.3, 0.7]])
        model.energy_and_gradient(descriptors, symbols)
        analytical = model.double_backward(seed)
        step = 1.0e-6
        for element in model.elements:
            weight = model.networks[element].weights[0]
            original = weight[0, 0]
            weight[0, 0] = original + step
            _, plus_gradient = model.energy_and_gradient(descriptors, symbols)
            plus = np.sum(seed * plus_gradient)
            weight[0, 0] = original - step
            _, minus_gradient = model.energy_and_gradient(descriptors, symbols)
            minus = np.sum(seed * minus_gradient)
            weight[0, 0] = original
            finite_difference = (plus - minus) / (2.0 * step)
            assert abs(
                analytical[element]["weights"][0][0, 0] - finite_difference
            ) < 1.0e-5
