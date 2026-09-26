import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from .descriptors import compute_descriptors
from .kernels import compute_2b_kernel_matrices, compute_3b_kernel_matrices


class GP:

    def __init__(
        self,
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
        self.r_cut_2b = r_cut_2b
        self.r_cut_3b = r_cut_3b
        self.jitter = jitter
        self.use_2b = use_2b
        self.use_3b = use_3b

        self._hypers = np.array(
            [sigma_2b, length_2b, sigma_3b, length_3b, noise], dtype=np.float64
        )

        self.frames = None
        self.nframes = 0
        self.natoms_list = None
        self.natoms_tot = 0

        self.desc = {}
        self.inducing_2b = None
        self.inducing_3b = None
        self.y_data = None

        self._Kmm = None
        self._Knm = None
        self.L = None
        self.alpha = None

    @property
    def hypers(self):
        return self._hypers.copy()

    @hypers.setter
    def hypers(self, v):
        self._hypers = np.array(v, dtype=np.float64)

    def _select_inducing(self, desc):
        raise NotImplementedError

    def _build_kernels(self, sigma2, sigma3, length2, length3, desc, nf, nat, inducing_2b, inducing_3b, use_energy=False, store_species_order=False):
        use_2b = self.use_2b and desc["body2_features"].size > 0
        use_3b = self.use_3b and desc["body3_features"].size > 0

        Kmm_blocks = []
        Knm_parts = []

        if use_2b:
            sp2 = desc["body2_features"][inducing_2b]
            sp2_species = desc["body2_species"][inducing_2b]
            if store_species_order:
                self._species_2b_order = sorted(set(desc["body2_species"]) | set(sp2_species))
            Kmm_b2, _, _, KnmE_b2, KnmF_b2 = compute_2b_kernel_matrices(
                sigma2, length2, self.r_cut_2b, nf, nat,
                desc["body2_features"], desc["body2_gradients"],
                desc["body2_mapping"], desc["body2_species"],
                sp2, sp2_species,
                allowed_species=getattr(self, '_species_2b_order', None),
            )
            Kmm_blocks.append(Kmm_b2)
            if use_energy:
                Knm_parts.append(np.hstack([KnmE_b2, KnmF_b2]))
            else:
                Knm_parts.append(KnmF_b2)

        if use_3b:
            sp3 = desc["body3_features"][inducing_3b]
            sp3_species = desc["body3_species"][inducing_3b]
            if store_species_order:
                self._species_3b_order = sorted(set(desc["body3_species"]) | set(sp3_species))
            Kmm_b3, _, _, KnmE_b3, KnmF_b3 = compute_3b_kernel_matrices(
                sigma3, length3, self.r_cut_3b, nf, nat,
                desc["body3_features"], desc["body3_gradients"],
                desc["body3_mapping"], desc["body3_species"],
                sp3, sp3_species,
                allowed_species=getattr(self, '_species_3b_order', None),
            )
            Kmm_blocks.append(Kmm_b3)
            if use_energy:
                Knm_parts.append(np.hstack([KnmE_b3, KnmF_b3]))
            else:
                Knm_parts.append(KnmF_b3)

        if not Kmm_blocks:
            raise RuntimeError("No descriptors available")

        if len(Kmm_blocks) == 1:
            Kmm = Kmm_blocks[0]
        else:
            dims = [b.shape[0] for b in Kmm_blocks]
            Kmm = np.zeros((sum(dims), sum(dims)))
            off = 0
            for b in Kmm_blocks:
                n = b.shape[0]
                Kmm[off:off + n, off:off + n] = b
                off += n

        Knm = np.hstack(Knm_parts)
        n_train = Knm.shape[0]

        jitter_mat = self.jitter * np.eye(Kmm.shape[0])
        try:
            L_mm = cholesky(Kmm + jitter_mat, lower=True)
            Kinv_KnmT = solve_triangular(
                L_mm.T, solve_triangular(L_mm, Knm.T, lower=True)
            )
        except np.linalg.LinAlgError:
            Kmm_reg = Kmm + jitter_mat
            Kinv_KnmT = np.linalg.solve(Kmm_reg, Knm.T)
        K_eff = Knm @ Kinv_KnmT + self._hypers[4] ** 2 * np.eye(n_train)

        return K_eff, Kmm, Knm

    def fit(self, frames, forces, energies=None):
        self.frames = list(frames)
        self.nframes = len(frames)
        self.natoms_list = np.array([len(a) for a in frames])
        self.natoms_tot = np.sum(self.natoms_list)

        self.desc = compute_descriptors(frames, self.r_cut_2b, self.r_cut_3b)
        self._select_inducing(self.desc)

        y_parts = []
        if energies is not None:
            y_parts.append(np.array(energies, dtype=np.float64))
        if forces is not None:
            y_parts.append(
                np.vstack([f.reshape(-1, 3) for f in forces]).ravel().astype(np.float64)
            )
        self.y_data = np.concatenate(y_parts) if y_parts else None

        h = self._hypers
        K_eff, Kmm, Knm = self._build_kernels(
            h[0], h[2], h[1], h[3],
            self.desc, self.nframes, self.natoms_tot,
            self.inducing_2b, self.inducing_3b,
            use_energy=energies is not None,
            store_species_order=True,
        )
        self._Kmm = Kmm
        self._Knm = Knm

        self.L = cholesky(K_eff, lower=True)
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y_data, lower=True)
        )

    def predict(self, test_frames, return_std=True):
        desc_test = compute_descriptors(test_frames, self.r_cut_2b, self.r_cut_3b)
        nat_tot = sum(len(a) for a in test_frames)
        nf_test = len(test_frames)

        h = self._hypers
        s2, l2, s3, l3, noise = h

        Knm_parts = []

        if self.use_2b and desc_test["body2_features"].size > 0 and self.desc["body2_features"].size > 0:
            sp2 = self.desc["body2_features"][self.inducing_2b]
            sp2_species = self.desc["body2_species"][self.inducing_2b]
            _, _, _, _, KnmF = compute_2b_kernel_matrices(
                s2, l2, self.r_cut_2b, nf_test, nat_tot,
                desc_test["body2_features"], desc_test["body2_gradients"],
                desc_test["body2_mapping"], desc_test["body2_species"],
                sp2, sp2_species,
                allowed_species=getattr(self, '_species_2b_order', None),
            )
            Knm_parts.append(KnmF)

        if self.use_3b and desc_test["body3_features"].size > 0 and self.desc["body3_features"].size > 0:
            sp3 = self.desc["body3_features"][self.inducing_3b]
            sp3_species = self.desc["body3_species"][self.inducing_3b]
            _, _, _, _, KnmF = compute_3b_kernel_matrices(
                s3, l3, self.r_cut_3b, nf_test, nat_tot,
                desc_test["body3_features"], desc_test["body3_gradients"],
                desc_test["body3_mapping"], desc_test["body3_species"],
                sp3, sp3_species,
                allowed_species=getattr(self, '_species_3b_order', None),
            )
            Knm_parts.append(KnmF)

        Knm_cross = np.hstack(Knm_parts)
        Kmm = self._Kmm

        jitter_mat = self.jitter * np.eye(Kmm.shape[0])
        L_mm = cholesky(Kmm + jitter_mat, lower=True)

        K_mm_inv_Knm_train_T = solve_triangular(
            L_mm.T, solve_triangular(L_mm, self._Knm.T, lower=True)
        )
        K_cross_val = Knm_cross @ K_mm_inv_Knm_train_T
        pred_mean = K_cross_val @ self.alpha

        if return_std:
            Kinv_KnmT_cross = solve_triangular(
                L_mm.T, solve_triangular(L_mm, Knm_cross.T, lower=True)
            )
            K_eff_test = Knm_cross @ Kinv_KnmT_cross + noise ** 2 * np.eye(Knm_cross.shape[0])
            v_cross = solve_triangular(self.L, K_cross_val.T, lower=True)
            pred_var = np.diag(K_eff_test) - np.sum(v_cross ** 2, axis=0)
            return pred_mean, np.sqrt(np.maximum(pred_var, 0))

        return pred_mean

    def predict_energy(self, test_frames, return_std=True):
        desc_test = compute_descriptors(test_frames, self.r_cut_2b, self.r_cut_3b)
        nat_tot = sum(len(a) for a in test_frames)
        nf_test = len(test_frames)

        h = self._hypers
        s2, l2, s3, l3 = h[:4]

        KnmE_parts = []

        if self.use_2b and desc_test["body2_features"].size > 0 and self.desc["body2_features"].size > 0:
            sp2 = self.desc["body2_features"][self.inducing_2b]
            sp2_species = self.desc["body2_species"][self.inducing_2b]
            _, _, _, KnmE_b2, _ = compute_2b_kernel_matrices(
                s2, l2, self.r_cut_2b, nf_test, nat_tot,
                desc_test["body2_features"], desc_test["body2_gradients"],
                desc_test["body2_mapping"], desc_test["body2_species"],
                sp2, sp2_species,
                allowed_species=getattr(self, '_species_2b_order', None),
            )
            KnmE_parts.append(KnmE_b2)

        if self.use_3b and desc_test["body3_features"].size > 0 and self.desc["body3_features"].size > 0:
            sp3 = self.desc["body3_features"][self.inducing_3b]
            sp3_species = self.desc["body3_species"][self.inducing_3b]
            _, _, _, KnmE_b3, _ = compute_3b_kernel_matrices(
                s3, l3, self.r_cut_3b, nf_test, nat_tot,
                desc_test["body3_features"], desc_test["body3_gradients"],
                desc_test["body3_mapping"], desc_test["body3_species"],
                sp3, sp3_species,
                allowed_species=getattr(self, '_species_3b_order', None),
            )
            KnmE_parts.append(KnmE_b3)

        KnmE_cross = np.hstack(KnmE_parts)
        Kmm = self._Kmm

        jitter_mat = self.jitter * np.eye(Kmm.shape[0])
        L_mm = cholesky(Kmm + jitter_mat, lower=True)

        K_mm_inv_Knm_train_T = solve_triangular(
            L_mm.T, solve_triangular(L_mm, self._Knm.T, lower=True)
        )
        K_cross_energy = KnmE_cross @ K_mm_inv_Knm_train_T
        pred_energy = K_cross_energy @ self.alpha

        if return_std:
            Kinv_KnmE_T = solve_triangular(
                L_mm.T, solve_triangular(L_mm, KnmE_cross.T, lower=True)
            )
            K_eff_test = KnmE_cross @ Kinv_KnmE_T
            v_cross = solve_triangular(self.L, K_cross_energy.T, lower=True)
            pred_var = np.diag(K_eff_test) - np.sum(v_cross ** 2, axis=0)
            return pred_energy, np.sqrt(np.maximum(pred_var, 0))

        return pred_energy

    def log_marginal_likelihood(self, hypers):
        h_old = self._hypers.copy()
        self._hypers = np.array(hypers)

        use_energy = self.y_data is not None and self.y_data.shape[0] > self.natoms_tot * 3

        try:
            K_eff, _, _ = self._build_kernels(
                hypers[0], hypers[2], hypers[1], hypers[3],
                self.desc, self.nframes, self.natoms_tot,
                self.inducing_2b, self.inducing_3b,
                use_energy=use_energy,
            )
            L = cholesky(K_eff, lower=True)
            alpha = solve_triangular(L.T, solve_triangular(L, self.y_data, lower=True))
            n = len(self.y_data)
            lml = -0.5 * self.y_data @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        except (np.linalg.LinAlgError, ValueError):
            lml = -np.inf
        finally:
            self._hypers = h_old

        return lml

    def optimize(self, maxiter=100):
        def neg_lml(h):
            return -self.log_marginal_likelihood(h)

        bounds = [
            (1e-6, 1e2), (1e-3, 1e2),
            (1e-6, 1e2), (1e-3, 1e2),
            (1e-6, 1e1),
        ]

        result = minimize(
            neg_lml,
            self._hypers,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-4, "gtol": 1e-4},
        )

        if result.success:
            self._hypers = result.x
            use_energy = self.y_data is not None and self.y_data.shape[0] > self.natoms_tot * 3
            K_eff, Kmm, Knm = self._build_kernels(
                result.x[0], result.x[2], result.x[1], result.x[3],
                self.desc, self.nframes, self.natoms_tot,
                self.inducing_2b, self.inducing_3b,
                use_energy=use_energy,
            )
            self._Kmm = Kmm
            self._Knm = Knm
            self.L = cholesky(K_eff, lower=True)
            self.alpha = solve_triangular(
                self.L.T, solve_triangular(self.L, self.y_data, lower=True)
            )

        return result

    def set_hyperparameters(self, sigma_2b=None, length_2b=None, sigma_3b=None, length_3b=None, noise=None):
        if sigma_2b is not None:
            self._hypers[0] = sigma_2b
        if length_2b is not None:
            self._hypers[1] = length_2b
        if sigma_3b is not None:
            self._hypers[2] = sigma_3b
        if length_3b is not None:
            self._hypers[3] = length_3b
        if noise is not None:
            self._hypers[4] = noise
