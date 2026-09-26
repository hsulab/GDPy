import numpy as np

from .base import GP


def _inducing_uniform(features, n_inducing):
    n = features.shape[0]
    if n <= n_inducing:
        return np.arange(n)
    indices = np.linspace(0, n - 1, n_inducing, dtype=int)
    return indices


class SGP(GP):

    def __init__(
        self,
        n_inducing=100,
        r_cut_2b=6.0,
        r_cut_3b=4.0,
        sigma_2b=1.0,
        length_2b=1.0,
        sigma_3b=1.0,
        length_3b=1.0,
        noise=0.01,
        jitter=1e-8,
        use_2b=True,
        use_3b=False,
    ):
        super().__init__(
            r_cut_2b=r_cut_2b,
            r_cut_3b=r_cut_3b,
            sigma_2b=sigma_2b,
            length_2b=length_2b,
            sigma_3b=sigma_3b,
            length_3b=length_3b,
            noise=noise,
            jitter=jitter,
            use_2b=use_2b,
            use_3b=use_3b,
        )
        self.n_inducing = n_inducing

    def _select_inducing(self, desc):
        n2 = desc["body2_features"].shape[0]
        n3 = desc["body3_features"].shape[0]
        total = (n2 if self.use_2b else 0) + (n3 if self.use_3b else 0)
        n_i2 = max(0, int(self.n_inducing * n2 / max(total, 1))) if self.use_2b else 0
        n_i3 = max(0, self.n_inducing - n_i2) if self.use_3b else 0
        self.inducing_2b = _inducing_uniform(desc["body2_features"], n_i2)
        self.inducing_3b = _inducing_uniform(desc["body3_features"], n_i3)
