#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from .builder import StructureModifier 


class PerturbatorBuilder(StructureModifier):

    name = "perturbater"

    """Perturb positions of input structures.

    TODO:
        1. Perturb cell.
        2. Perturb distances, angles...
        3. Check if perturbed structures are valid (too close distance).

    """

    def __init__(self, eps: float=0.2, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.eps = eps # unit Ang

        return
    
    def run(self, substrates: List[Atoms]=None, size:int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate, size)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int):
        """"""
        frames = []
        for i in range(size):
            atoms = copy.deepcopy(substrate)
            natoms = len(atoms)
            pos_drift = self.rng.random((natoms,3))
            atoms.positions += pos_drift*self.eps
            frames.append(atoms)

        return frames


if __name__ == "__main__":
    ...