#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Mapping

import numpy as np
import numpy.typing
from ase import Atoms
from joblib import Parallel, delayed

#: The name of the fingerprint.
FINGERPRINT_NAME = "interatomic_distance"


def get_sorted_dist_list(atoms: Atoms, mic: bool = False) -> Mapping[int, numpy.typing.NDArray]:
    """calculate the sorted distance list describing the cluster in atoms.

    Args:
        atoms: The input structure.
        mic: Whether use mic for distances.

    """
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = {}
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]  # type: ignore
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1 :]:
                d.append(atoms.get_distance(n1, n2, mic))
        d.sort()
        pair_cor[n] = np.array(d)

    return pair_cor


class InteratomicDistanceComparator:
    """Compare two atoms based on inter-atomic distances.

    An implementation of the comparison criteria described in L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012).

    """

    def __init__(self, n_top=None, pair_cor_cum_diff=0.015, pair_cor_max=0.7, dE=0.02, mic=False, n_jobs=1):
        """

        Args:
            n_top: The number of atoms being optimized by the GA.
                Default 0 - meaning all atoms.

            pair_cor_cum_diff: The limit in eq. 2 of the letter.
            pair_cor_max: The limit in eq. 3 of the letter
            dE: The limit of eq. 1 of the letter
            mic: Determines if distances are calculated
            using the minimum image convention

        """
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.mic = mic

        self.n_jobs = n_jobs

        return

    def _precompute_fingerprint(self, frames: list[Atoms]) -> None:
        """Precompute the fingerprints for all structures in the list.

        The fingerprint is stored in the `info` dictionary of each `Atoms` object.

        """
        fingerprints = Parallel(n_jobs=self.n_jobs)(
            delayed(get_sorted_dist_list)(atoms, mic=self.mic) for atoms in frames
        )
        for atoms, fingerprint in zip(frames, fingerprints):
            atoms.info[FINGERPRINT_NAME] = fingerprint

        return

    def _delete_fingerprint(self, frames: list[Atoms]) -> None:
        """Delete the fingerprints from all structures in the list.

        This is useful to free memory after comparison.

        """
        for atoms in frames:
            if FINGERPRINT_NAME in atoms.info:
                del atoms.info[FINGERPRINT_NAME]

        return

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """Return if structure a1 or a2 are similar or not.

        Args:
            a1: The first structure.
            a2: The second structure.

        Returns:
            Whether the two structures look like each other.

        """
        # Check if two structures have consistent atoms.
        if len(a1) != len(a2):
            return False
        else:
            if a1.get_chemical_symbols() != a2.get_chemical_symbols():
                return False

        # first we check the energy criteria
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False

        # then we check the structure
        a1top = a1[-self.n_top :]
        a2top = a2[-self.n_top :]
        cum_diff, max_diff = self.__compare_structure__(a1top, a2top)  # type: ignore

        return cum_diff < self.pair_cor_cum_diff and max_diff < self.pair_cor_max

    def __compare_structure__(self, a1: Atoms, a2: Atoms):
        """Private method for calculating the structural difference."""
        # Get the sorted distance list for both structures.
        # Use cache to avoid recalculating the same fingerprint.
        if FINGERPRINT_NAME not in a1.info:
            p1 = get_sorted_dist_list(a1, mic=self.mic)
            a1.info[FINGERPRINT_NAME] = p1
        else:
            p1 = a1.info[FINGERPRINT_NAME]
        if FINGERPRINT_NAME not in a2.info:
            p2 = get_sorted_dist_list(a2, mic=self.mic)
            a2.info[FINGERPRINT_NAME] = p2
        else:
            p2 = a2.info[FINGERPRINT_NAME]

        # Compare the two fingerprints.
        numbers = a1.numbers
        total_cum_diff = 0.0
        max_diff = 0
        for n in p1.keys():
            cum_diff = 0.0
            c1 = p1[n]
            c2 = p2[n]
            assert len(c1) == len(c2)
            if len(c1) == 0:
                continue
            t_size = np.sum(c1)
            d = np.abs(c1 - c2)
            cum_diff = np.sum(d)
            max_diff = np.max(d)
            ntype = float(sum(i == n for i in numbers))
            total_cum_diff += cum_diff / t_size * ntype / float(len(numbers))

        return (total_cum_diff, max_diff)


if __name__ == "__main__":
    ...
