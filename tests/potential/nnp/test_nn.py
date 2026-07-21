import numpy as np
from gdpx.potential.nnp.nn import SimpleNN


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
