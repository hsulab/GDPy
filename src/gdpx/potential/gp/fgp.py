import numpy as np

from .base import GP


class FGP(GP):

    def __init__(
        self,
        r_cut_2b=6.0,
        r_cut_3b=4.0,
        sigma_2b=1.0,
        length_2b=1.0,
        sigma_3b=1.0,
        length_3b=1.0,
        noise=0.01,
        jitter=1e-4,
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

    def _select_inducing(self, desc):
        self.inducing_2b = np.arange(desc["body2_features"].shape[0]) if self.use_2b else np.array([], dtype=int)
        self.inducing_3b = np.arange(desc["body3_features"].shape[0]) if self.use_3b else np.array([], dtype=int)
