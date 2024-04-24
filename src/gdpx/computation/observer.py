#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

from typing import List

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.ga.utilities import closest_distances_generator

from ..core.register import registers, Register

registers.observer = Register("observer")


class Observer:

    def __init__(self, patience: int = 0, *args, **kwargs):
        """"""
        self._patience = patience

        return

    @property
    def patience(self) -> int:
        """"""

        return self._patience


class DistanceObserver(Observer):

    def __init__(
        self,
        symbols: List[str],
        covalent_ratio=[0.8, 2.0],
        custom_dmin=[],
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.symbols = symbols

        self.cov_min, self.cov_max = covalent_ratio

        custom_dmin_dict = {}
        for i, j, d in custom_dmin:
            s_i, s_j = atomic_numbers[i], atomic_numbers[j]
            custom_dmin_dict[(s_i, s_j)] = d
            custom_dmin_dict[(s_j, s_i)] = d
        self.custom_dmin_dict = custom_dmin_dict

        self._neighlist = None

        return

    def run(self, atoms: Atoms):
        """"""
        should_stop = False

        # FIXME: We assume the input atoms are from AseDriver and are always the same,
        #        and every simulation will init a new observer.
        #        So we are safe right now but we need compare atoms in the future.
        if self._neighlist is None:
            self._neighlist = NeighborList(
                self.cov_min * np.array(natural_cutoffs(atoms)),
                skin=0.0,
                self_interaction=False,
                bothways=True,
            )
            self._neighlist.update(atoms)

            self._dmin_dict = closest_distances_generator(
                set(atoms.get_atomic_numbers()), self.cov_min
            )
            self._dmin_dict.update(self.custom_dmin_dict)
            selected_indices = [
                i for i, a in enumerate(atoms) if a.symbol in self.symbols
            ]
            self._included_pairs = list(itertools.permutations(selected_indices, 2))
        else:
            ...

        if not self._check_whether_distance_is_not_small(
            atoms, self._neighlist, self._dmin_dict, included_pairs=self._included_pairs
        ):
            should_stop = True

        return should_stop

    def _check_whether_distance_is_not_small(
        self, atoms: Atoms, neighlist, dmin_dict, included_pairs=[], excluded_pairs=[]
    ):
        """use neighbour list to check newly added atom is neither too close or too
        far from other atoms
        """
        atomic_numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell(complete=True)
        natoms = len(atoms)

        is_valid = True
        for i in range(natoms):
            nei_indices, nei_offsets = neighlist.get_neighbors(i)
            for j, offset in zip(nei_indices, nei_offsets):
                if (i, j) in included_pairs and (i, j) not in excluded_pairs:
                    print(f"check {i} <> {j}")
                    distance = np.linalg.norm(
                        atoms.positions[i] - (atoms.positions[j] + np.dot(offset, cell))
                    )
                    atomic_pair = (atomic_numbers[i], atomic_numbers[j])
                    if distance < dmin_dict[atomic_pair]:
                        is_valid = False
                        break

        return is_valid


registers.observer.register("distance")(DistanceObserver)


def create_an_observer(params: dict) -> "Observer":
    """"""
    params = copy.deepcopy(params)
    method = params.pop("method")

    observer = registers.observer[method](**params)

    return observer


if __name__ == "__main__":
    ...
