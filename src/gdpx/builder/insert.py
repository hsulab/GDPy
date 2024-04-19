#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

from typing import Optional, List, Tuple

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic

from . import registers
from .builder import StructureModifier
from .utils import (
    check_overlap_neighbour,
    convert_composition_to_list,
    rotate_a_molecule,
)


class InsertModifier(StructureModifier):

    name = "insert"

    def __init__(
        self,
        region,
        composition: dict,
        custom_dmin: list = [],
        covalent_ratio=[0.8, 2.0],
        molecular_distol=[-np.inf, np.inf],
        max_times_size: int = 5,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        # - system definition
        self.region = copy.deepcopy(region)  # TODO: str, dict, or a Region obeject?
        shape = region.pop("method", None)
        self._region = registers.create("region", shape, convert_name=True, **region)

        self.composition = composition
        self._composition_list = convert_composition_to_list(composition, self._region)

        for k, v in self._composition_list:
            if len(k) > 1:
                self._insert_molecule = True
                break
        else:
            self._insert_molecule = False

        self.intermol_dismin = molecular_distol[0]
        self.intermol_dismax = molecular_distol[1]

        custom_dmin_dict = {}
        for i, j, d in custom_dmin:
            s_i, s_j = atomic_numbers[i], atomic_numbers[j]
            custom_dmin_dict[(s_i, s_j)] = d
            custom_dmin_dict[(s_j, s_i)] = d
        self.custom_dmin_dict = custom_dmin_dict

        # - bond distance check
        self.covalent_ratio = covalent_ratio
        self.MAX_TIMES_SIZE = max_times_size

        return

    def run(
        self, substrates: List[Atoms] = None, size: int = 1, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        # If no substrates, create a empty box with the region as its cell TODO:
        _substrates = self.substrates
        if _substrates is None:
            _substrates = [Atoms()]

        frames = []
        for substrate in _substrates:
            curr_frames = self._irun(substrate, size)
            frames.extend(curr_frames)

        return frames

    def _irun(self, substrate: Atoms, size: int):
        """"""
        frames = []
        for i in range(size * self.MAX_TIMES_SIZE):
            nframes = len(frames)
            if nframes < size:
                atoms = copy.deepcopy(substrate)
                excluded_pairs = list(itertools.permutations(range(len(atoms)), 2))
                atoms = self._insert_species(atoms, excluded_pairs)
                if atoms is not None:
                    frames.append(atoms)
                    self._print(f"{nframes =} at {i}")
            else:
                break
        else:
            raise RuntimeError(
                f"Failed to create {size} structures, only {nframes} are created."
            )

        return frames

    def _insert_species(
        self, substrate: Atoms, excluded_pairs: List[Tuple[int, int]]
    ) -> Optional[Atoms]:
        """"""
        atoms = substrate
        atoms.set_tags(0)

        species_list = itertools.chain(
            *[[k for i in range(v)] for k, v in self._composition_list]
        )
        species_list = sorted(species_list, key=lambda a: a.get_chemical_formula())
        num_species = len(species_list)

        if not self._insert_molecule:
            random_positions = self._region.get_random_positions(
                size=num_species, rng=self.rng
            )
        else:
            for i in range(self.MAX_TIMES_SIZE):
                random_positions = self._region.get_random_positions(
                    size=num_species, rng=self.rng
                )
                pair_positions = np.array(
                    list(itertools.combinations(random_positions, 2))
                )
                raw_vectors = pair_positions[:, 0, :] - pair_positions[:, 1, :]
                mic_vecs, mic_dis = find_mic(v=raw_vectors, cell=atoms.cell)
                if np.min(mic_dis) >= self.intermol_dismin:
                    break
            else:
                raise RuntimeError(
                    f"Failed to get molecular positions after {self.MAX_TIMES_SIZE} attempts."
                )

        intra_bonds = []
        for a, p in zip(species_list, random_positions):
            # count number of atoms
            prev_num_atoms = len(atoms)
            curr_num_atoms = prev_num_atoms + len(a)
            intra_bonds.extend(
                list(itertools.permutations(range(prev_num_atoms, curr_num_atoms), 2))
            )
            # rotate and translate
            a = rotate_a_molecule(a, use_com=True, rng=self.rng)
            # a.translate(p - np.mean(a.positions, axis=0))
            a.translate(p - a.get_center_of_mass())
            a.set_tags(int(np.max(atoms.get_tags()) + 1))
            atoms += a

        excluded_pairs.extend(intra_bonds)
        if check_overlap_neighbour(
            atoms,
            covalent_ratio=self.covalent_ratio,
            custom_dmin_dict=self.custom_dmin_dict,
            excluded_pairs=excluded_pairs,
        ):
            ...
        else:
            atoms = None

        return atoms


if __name__ == "__main__":
    ...
