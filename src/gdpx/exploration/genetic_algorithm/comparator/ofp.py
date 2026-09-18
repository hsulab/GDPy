"""GDPy implementation of the ASE-GA 1.0.3-style OFP comparator."""

from __future__ import annotations

import numpy as np
from ase.neighborlist import neighbor_list
from ase.utils import pbc2pbc


class OFPComparator:
    """Compare structures using Gaussian-smeared pair-distance fingerprints.

    Constructor parameters intentionally match ASE-GA's public comparator.
    """

    def __init__(
        self,
        n_top=None,
        dE=1.0,
        cos_dist_max=5e-3,
        rcut=20.0,
        binwidth=0.05,
        sigma=0.02,
        nsigma=4,
        pbc=True,
        maxdims=None,
        recalculate=False,
    ):
        self.n_top = n_top or 0
        self.dE = dE
        self.cos_dist_max = cos_dist_max
        self.rcut = rcut
        self.binwidth = binwidth
        self.sigma = sigma
        self.nsigma = nsigma
        self.pbc = pbc2pbc(pbc)
        self.maxdims = [None] * 3 if maxdims is None else maxdims
        self.recalculate = recalculate

    def _fingerprint(self, atoms) -> tuple[np.ndarray, tuple[int, ...]]:
        candidate = atoms[-self.n_top :] if self.n_top else atoms
        numbers = candidate.get_atomic_numbers()
        types = tuple(sorted(set(int(number) for number in numbers)))
        bins = max(1, int(np.ceil(self.rcut / self.binwidth)))
        parts = []
        first, second, distances = neighbor_list("ijd", candidate, self.rcut, self_interaction=False)
        for type_index, first_type in enumerate(types):
            for second_type in types[type_index:]:
                selected = distances[
                    ((numbers[first] == first_type) & (numbers[second] == second_type))
                    | ((numbers[first] == second_type) & (numbers[second] == first_type))
                ]
                histogram, _ = np.histogram(selected, bins=bins, range=(0.0, self.rcut))
                if self.sigma > 0:
                    radius = max(1, int(np.ceil(self.nsigma * self.sigma / self.binwidth)))
                    x = np.arange(-radius, radius + 1) * self.binwidth
                    kernel = np.exp(-0.5 * (x / self.sigma) ** 2)
                    kernel /= kernel.sum()
                    histogram = np.convolve(histogram.astype(float), kernel, mode="same")
                parts.append(histogram.astype(float))
        return np.concatenate(parts) if parts else np.zeros(bins), types

    def _compare_structure(self, first, second) -> float:
        fp_first, types_first = self._fingerprint(first)
        fp_second, types_second = self._fingerprint(second)
        if types_first != types_second or fp_first.shape != fp_second.shape:
            raise AssertionError("The two structures have different compositions.")
        denominator = np.linalg.norm(fp_first) * np.linalg.norm(fp_second)
        if denominator == 0:
            return 0.0 if np.array_equal(fp_first, fp_second) else 1.0
        return float(0.5 * (1.0 - np.dot(fp_first, fp_second) / denominator))

    def looks_like(self, first, second) -> bool:
        if len(first) != len(second):
            raise ValueError("The two configurations are not the same size.")
        if first.calc is not None and second.calc is not None:
            if abs(first.get_potential_energy() - second.get_potential_energy()) >= self.dE:
                return False
        return self._compare_structure(first, second) < self.cos_dist_max
