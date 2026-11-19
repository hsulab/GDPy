from typing import Mapping

import numpy as np
import numpy.typing
from ase import Atoms
from ase.geometry import find_mic
from joblib import Parallel, delayed

#: The name of the fingerprint.
FINGERPRINT_NAME = "interatomic_distance"

# Bound the temporary arrays used by ASE for periodic image searches.
_DISTANCE_BATCH_SIZE = 16384


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
        positions = atoms.positions[numbers == n]
        first, second = np.triu_indices(len(positions), k=1)
        distances = np.empty(len(first), dtype=float)
        for start in range(0, len(first), _DISTANCE_BATCH_SIZE):
            stop = start + _DISTANCE_BATCH_SIZE
            vectors = positions[second[start:stop]] - positions[first[start:stop]]
            if mic:
                _, lengths = find_mic(vectors, atoms.cell, atoms.pbc)
            else:
                lengths = np.linalg.norm(vectors, axis=1)
            distances[start:stop] = lengths
        distances.sort()
        pair_cor[n] = distances

    return pair_cor


class InteratomicDistanceComparator:
    """Compare two atoms based on inter-atomic distances.

    An implementation of the comparison criteria described in L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012).

    """

    def __init__(self, pair_cor_cum_diff=0.015, pair_cor_max=0.7, dE=0.02, mic=False, n_jobs=1):
        """

        Args:
            pair_cor_cum_diff: The limit in eq. 2 of the letter.
            pair_cor_max: The limit in eq. 3 of the letter
            dE: The limit of eq. 1 of the letter
            mic: Determines if distances are calculated
            using the minimum image convention

        """
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
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
        cum_diff, max_diff = self.__compare_structure__(a1, a2)

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
