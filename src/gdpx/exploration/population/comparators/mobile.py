"""Population-owned selection of mobile atoms for structural comparison."""

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.analysis.comparators.comparator import BaseComparator
from gdpx.structures.groups.group import evaluate_group_expression


class MobileComparator(BaseComparator):
    """Compare tag != 0 atoms, optionally narrowed by an original-index group.

    Selected copies and their fingerprints are owned by this comparator. Energy
    gates use complete candidate energies, never a subset energy calculation.
    """

    def __init__(self, comparator, group_expression=None, preserve_energy=True):
        self.comparator = comparator
        self.group_expression = group_expression
        self.preserve_energy = preserve_energy
        self._prepared = {}

    def _select(self, atoms):
        selected = set(evaluate_group_expression(atoms, self.group_expression))
        indices = [i for i, tag in enumerate(atoms.get_tags()) if tag != 0 and i in selected]
        if not indices:
            raise ValueError("Population comparison selected no mobile atoms (tag != 0).")
        candidate = atoms[indices]
        # Do not reuse a fingerprint computed for another group or the full slab.
        candidate.info = {}
        if self.preserve_energy and atoms.calc is not None:
            if len(indices) == len(atoms):
                candidate.calc = atoms.calc
            else:
                candidate.calc = SinglePointCalculator(candidate, energy=atoms.get_potential_energy())
        return candidate

    def _candidate(self, atoms):
        entry = self._prepared.get(id(atoms))
        return entry[1] if entry is not None and entry[0] is atoms else self._select(atoms)

    def _precompute_fingerprint(self, frames):
        selected = [self._select(atoms) for atoms in frames]
        for atoms, candidate in zip(frames, selected):
            self._prepared[id(atoms)] = (atoms, candidate)
        if hasattr(self.comparator, "_precompute_fingerprint"):
            self.comparator._precompute_fingerprint(selected)

    def _delete_fingerprint(self, frames):
        for atoms in frames:
            self._prepared.pop(id(atoms), None)

    def looks_like(self, first, second):
        left, right = self._candidate(first), self._candidate(second)
        if not np.array_equal(np.sort(left.numbers), np.sort(right.numbers)):
            return False
        return self.comparator.looks_like(left, right)
