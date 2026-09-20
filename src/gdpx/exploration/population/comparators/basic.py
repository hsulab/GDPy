"""GDPy-owned basic and nearest-neighbor comparators."""

import numpy as np

from gdpx.structures.geometry.ga import get_nnmat


class AtomsComparator:
    """Compare ASE Atoms objects directly."""

    def looks_like(self, first, second) -> bool:
        return first == second


class NNMatComparator:
    """Compare normalized nearest-neighbor matrix fingerprints."""

    def __init__(self, d: float = 0.2, elements=None, mic: bool = False):
        self.d = d
        self.elements = [] if elements is None else elements
        self.mic = mic

    def looks_like(self, first, second) -> bool:
        elements = self.elements or sorted(set(first.get_chemical_symbols()))
        candidates = []
        for atoms in (first, second):
            candidate = atoms.copy()
            candidate.set_constraint()
            del candidate[[atom.index for atom in candidate if atom.symbol not in elements]]
            candidates.append(candidate)
        return bool(np.linalg.norm(get_nnmat(candidates[0], self.mic) - get_nnmat(candidates[1], self.mic)) < self.d)

