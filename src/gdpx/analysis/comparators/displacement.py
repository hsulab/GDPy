"""Atomic displacement comparison in the input coordinate frame."""

import numpy as np
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

from .comparator import BaseComparator


class DisplacementComparator(BaseComparator):
    """Compare RMS displacement with optional species matching and MIC.

    No translation or rotation fitting is performed. With permutation
    invariance, assignment minimizes the total squared displacement within
    each element. MIC is applied to every candidate pair before assignment.
    """

    name = "displacement"

    def __init__(self, group_expression=None, rmsd_max=0.5,
                 permutation_invariant=False, mic=False, **kwargs):
        if "n_top" in kwargs:
            raise ValueError("n_top is removed; use group_expression to select comparison atoms.")
        if "align" in kwargs:
            raise ValueError("align is not supported; displacement uses the input coordinate frame.")
        super().__init__(**kwargs)
        if not np.isfinite(rmsd_max) or rmsd_max <= 0:
            raise ValueError("rmsd_max must be positive and finite (Angstrom).")
        if not isinstance(permutation_invariant, bool) or not isinstance(mic, bool):
            raise ValueError("permutation_invariant and mic must be booleans.")
        self.group_expression = group_expression
        self.rmsd_max = rmsd_max
        self.permutation_invariant = permutation_invariant
        self.mic = mic

    def prepare_data(self, structures):
        # Retain input order: sorting by species would change fixed-index pairing.
        return self._prepare_groups(structures, canonicalize=False)

    def _distances(self, vectors, atoms):
        shape = vectors.shape[:-1]
        vectors = vectors.reshape(-1, 3)
        if self.mic and atoms.pbc.any():
            _, distances = find_mic(vectors, atoms.cell, pbc=atoms.pbc)
        else:
            distances = np.linalg.norm(vectors, axis=1)
        return distances.reshape(shape)

    def fingerprint_distance(self, first, second):
        """Return minimum RMS displacement of prepared groups in Angstrom."""
        if not len(first) or not len(second):
            raise ValueError("Displacement comparison requires nonempty groups.")
        if not np.array_equal(np.sort(first.numbers), np.sort(second.numbers)):
            raise ValueError("Comparison groups must have identical compositions.")
        if self.mic:
            if not np.array_equal(first.pbc, second.pbc):
                raise ValueError("MIC comparison requires identical periodic boundary conditions.")
            if first.pbc.any():
                if not np.allclose(first.cell, second.cell):
                    raise ValueError("MIC comparison requires identical cells.")
                if (not np.all(np.isfinite(first.cell)) or
                        np.any(np.linalg.norm(first.cell[first.pbc], axis=1) == 0)):
                    raise ValueError("MIC comparison requires finite, nonzero periodic cell vectors.")

        if self.permutation_invariant:
            distances = np.empty(len(first))
            for number in np.unique(first.numbers):
                left = np.flatnonzero(first.numbers == number)
                right = np.flatnonzero(second.numbers == number)
                pair_distances = self._distances(
                    first.positions[left, None, :] - second.positions[right, :], first
                )
                rows, columns = linear_sum_assignment(pair_distances ** 2)
                distances[left[rows]] = pair_distances[rows, columns]
        else:
            if not np.array_equal(first.numbers, second.numbers):
                raise ValueError("Fixed-index comparison requires matching species at each index.")
            distances = self._distances(first.positions - second.positions, first)
        return float(np.sqrt(np.mean(distances ** 2)))

    def compare_fingerprints(self, first, second):
        return self.fingerprint_distance(first, second) < self.rmsd_max

    def looks_like(self, first, second):
        return self.compare_fingerprints(*self.prepare_data([first, second]))
