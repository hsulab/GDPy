#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import List

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.ga.utilities import closest_distances_generator
from ase.neighborlist import NeighborList, natural_cutoffs

from ..core.register import Register, registers

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


class SmallDistanceObserver(Observer):

    def __init__(
        self,
        symbols: List[str],
        covalent_ratio=[0.8, 2.0],
        custom_dmax=[],
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

        custom_dmax_dict = {}
        for i, j, d in custom_dmax:
            s_i, s_j = atomic_numbers[i], atomic_numbers[j]
            custom_dmax_dict[(s_i, s_j)] = d
            custom_dmax_dict[(s_j, s_i)] = d
        self.custom_dmax_dict = custom_dmax_dict

        self._nei_ratio = self.cov_min
        self._neighlist = None

        return

    def run(self, atoms: Atoms):
        """"""
        # FIXME: We assume the input atoms are from AseDriver and are always the same,
        #        and every simulation will init a new observer.
        #        So we are safe right now but we need compare atoms in the future.
        if self._neighlist is None:
            self._neighlist = NeighborList(
                self._nei_ratio * np.array(natural_cutoffs(atoms)),
                skin=0.0,
                self_interaction=False,
                bothways=True,
            )

            self._dmin_dict = closest_distances_generator(
                set(atoms.get_atomic_numbers()), self.cov_min
            )
            self._dmin_dict.update(self.custom_dmin_dict)

            self._dmax_dict = closest_distances_generator(
                set(atoms.get_atomic_numbers()), self.cov_max
            )
            self._dmax_dict.update(self.custom_dmax_dict)

            selected_indices = [
                i for i, a in enumerate(atoms) if a.symbol in self.symbols
            ]
            self._included_indices = selected_indices
            self._included_pairs = list(itertools.permutations(selected_indices, 2))
        else:
            ...

        self._neighlist.update(atoms)

        should_stop = self._irun(atoms)

        return should_stop

    def _irun(self, atoms: Atoms) -> bool:
        """"""
        should_stop = False
        if not self._check_whether_distance_is_not_small(
            atoms, self._neighlist, self._dmin_dict, included_pairs=self._included_pairs
        ):
            should_stop = True

        return should_stop

    def _check_whether_distance_is_not_small(
        self, atoms: Atoms, neighlist, dmin_dict, included_pairs=[], excluded_pairs=[]
    ) -> bool:
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
                    distance = np.linalg.norm(
                        atoms.positions[i] - (atoms.positions[j] + np.dot(offset, cell))
                    )
                    atomic_pair = (atomic_numbers[i], atomic_numbers[j])
                    if distance < dmin_dict[atomic_pair]:
                        is_valid = False
                        break

        return is_valid


class IsolatedAtomObserver(SmallDistanceObserver):

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self._nei_ratio = self.cov_max

        return

    def _irun(self, atoms: Atoms) -> bool:
        """"""
        should_stop = False
        if self._check_whether_atom_is_isolated(
            atoms, self._neighlist, included_indices=self._included_indices
        ):
            should_stop = True

        return should_stop

    def _check_whether_atom_is_isolated(
        self, atoms: Atoms, neighlist, included_indices: List[int]
    ) -> bool:
        """use neighbour list to check newly added atom is neither too close or too
        far from other atoms
        """
        atomic_numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell(complete=True)
        natoms = len(atoms)

        is_isolated = False
        for i in range(natoms):
            if i in included_indices:
                nei_indices, nei_offsets = neighlist.get_neighbors(i)
                if len(nei_indices) == 0:
                    print(f"isolated {i}")
                    is_isolated = True

                # for j, offset in zip(nei_indices, nei_offsets):
                #     distance = np.linalg.norm(
                #         atoms.positions[i] - (atoms.positions[j] + np.dot(offset, cell))
                #     )
                #     print(f"check {i} <> {j} {distance}")

        return is_isolated


class MoleculeNumberObserver(Observer):

    def __init__(self, species: List[str], number_limit: int, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.species = species
        self.number_limit = number_limit

        return

    def run(self, atoms: Atoms):
        """"""
        should_stop = False

        tags = atoms.get_tags()
        tags_dict = {}  # species -> tag list
        for key, group in itertools.groupby(enumerate(tags), key=lambda x: x[1]):
            cur_indices = [x[0] for x in group]
            cur_atoms = atoms[cur_indices]
            formula = cur_atoms.get_chemical_formula()
            if formula not in tags_dict:
                tags_dict[formula] = []
            tags_dict[formula].append([key, cur_indices])

        curr_number = sum(len(tags_dict.get(s, [])) for s in self.species)
        if curr_number > self.number_limit:
            should_stop = True

        return should_stop


registers.observer.register("small_distance")(SmallDistanceObserver)
registers.observer.register("isolated_atom")(IsolatedAtomObserver)
registers.observer.register("molecule_number")(MoleculeNumberObserver)


def create_an_observer(params: dict) -> "Observer":
    """"""
    params = copy.deepcopy(params)
    method = params.pop("method")

    observer = registers.observer[method](**params)

    return observer


if __name__ == "__main__":
    ...
