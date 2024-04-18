#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

from typing import List

import numpy as np

from ase import Atoms

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
        covalent_ratio=[0.8, 2.0],
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
                atoms, intra_bonds = self._insert_species(atoms)
                excluded_pairs.extend(intra_bonds)
                if check_overlap_neighbour(
                    atoms, self.covalent_ratio, excluded_pairs=excluded_pairs
                ):
                    frames.append(atoms)
            else:
                break
        else:
            raise RuntimeError(
                f"Failed to create {size} structures, only {nframes} are created."
            )

        return frames

    def _insert_species(self, atoms: Atoms):
        """"""
        species_list = itertools.chain(
            *[[k for i in range(v)] for k, v in self._composition_list]
        )
        species_list = sorted(species_list, key=lambda a: a.get_chemical_formula())
        nspecies = len(species_list)

        random_positions = self._region.get_random_positions(
            size=nspecies, rng=self.rng
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
            a = rotate_a_molecule(a, rng=self.rng)
            a.translate(p - np.mean(a.positions, axis=0))
            atoms += a

        return atoms, intra_bonds


if __name__ == "__main__":
    ...
