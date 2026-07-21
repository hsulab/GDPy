import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .descriptor import compute_symmetry_functions
from .nn import SimpleNN


class ACSFNN(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, elements, g2_params, g4_params, g5_params, r_cut,
                 hidden_sizes=(64, 64), nn_weights=None,
                 fd_h=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.elements = list(elements)
        self.g2_params = list(g2_params)
        self.g4_params = list(g4_params)
        self.g5_params = list(g5_params)
        self.r_cut = float(r_cut)
        self.fd_h = float(fd_h)

        n_features = self._compute_n_features()
        self.nn = SimpleNN(n_features, hidden_sizes=hidden_sizes)
        if nn_weights is not None:
            self.nn.set_params(nn_weights)

    def _compute_n_features(self):
        n_elem = len(self.elements)
        n_g2 = len(self.g2_params)
        n_pairs = n_elem * (n_elem + 1) // 2
        n_g4 = len(self.g4_params)
        n_g5 = len(self.g5_params)
        return n_elem * n_g2 + n_pairs * (n_g4 + n_g5)

    def _compute_descriptor(self, atoms):
        return compute_symmetry_functions(atoms, self.elements, self.g2_params,
                                           self.g4_params, self.g5_params, self.r_cut)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        G = self._compute_descriptor(self.atoms)
        energy_per_atom, dE_dG = self.nn.energy_and_gradient(G)
        self.results["energy"] = float(np.sum(energy_per_atom))

        if "forces" in properties:
            pos0 = self.atoms.positions.copy()
            natoms = len(self.atoms)
            forces = np.zeros((natoms, 3))
            h = self.fd_h
            try:
                for ci in range(natoms):
                    for d in range(3):
                        self.atoms.positions[ci, d] = pos0[ci, d] + h
                        G_plus = self._compute_descriptor(self.atoms)
                        self.atoms.positions[ci, d] = pos0[ci, d] - h
                        G_minus = self._compute_descriptor(self.atoms)
                        self.atoms.positions[ci, d] = pos0[ci, d]
                        dG = (G_plus - G_minus) / (2.0 * h)
                        forces[ci, d] = -float(np.sum(dE_dG * dG))
            finally:
                self.atoms.positions[:] = pos0
            self.results["forces"] = forces
