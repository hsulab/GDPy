"""Comparison of total potential energies, independent of geometry."""

import numpy as np

from .comparator import BaseComparator


class EnergyComparator(BaseComparator):
    """Consider candidates similar when their total energy difference is small."""

    name = "energy"

    def __init__(self, ene_diff_tol=1.0, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(ene_diff_tol) or ene_diff_tol <= 0:
            raise ValueError("ene_diff_tol must be positive and finite (eV).")
        self.ene_diff_tol = ene_diff_tol

    def prepare_data(self, structures):
        """Read finite total potential energies from complete input structures."""
        energies = []
        for atoms in structures:
            energy = float(atoms.get_potential_energy())
            if not np.isfinite(energy):
                raise ValueError("Energy comparison requires finite potential energies.")
            energies.append(energy)
        return energies

    def compare_fingerprints(self, first, second):
        return abs(first - second) < self.ene_diff_tol

    def looks_like(self, first, second):
        return self.compare_fingerprints(*self.prepare_data([first, second]))
