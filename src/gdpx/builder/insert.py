#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

from typing import List

from ase import Atoms
from ase.io import read, write


from . import registers
from .builder import StructureModifier
from .utils import check_overlap_neighbour, convert_composition_to_list


class InsertModifier(StructureModifier):

    name = "insert"

    def __init__(
        self, region, composition: dict, covalent_ratio=[0.8, 2.0], 
        max_times_size: int=5, *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        # - system definition
        self.region = copy.deepcopy(region) # TODO: str, dict, or a Region obeject?
        shape = region.pop("method", None)
        self._region = registers.create("region", shape, convert_name=True, **region)

        self.composition = composition
        self._composition_list = convert_composition_to_list(composition, region)

        # - bond distance check
        self.covalent_ratio = covalent_ratio
        self.MAX_TIMES_SIZE = max_times_size

        return
    
    def run(self, substrates: List[Atoms]=None, size:int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        # TODO: if substrates is None?

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate, size)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int):
        """"""
        frames = []
        for i in range(size*self.MAX_TIMES_SIZE):
            nframes = len(frames)
            if nframes < size:
                atoms = copy.deepcopy(substrate)
                atoms = self._insert_species(atoms)
                if check_overlap_neighbour(atoms, self.covalent_ratio):
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
        for a, p in zip(species_list, random_positions):
            # TODO: molecule?
            a.positions = p
            atoms += a
        #write("xxx.xyz", atoms)
        #exit()

        return atoms


if __name__ == "__main__":
    ...