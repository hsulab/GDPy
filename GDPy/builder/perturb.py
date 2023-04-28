#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.register import registers
from GDPy.core.operation import Operation
from GDPy.builder.builder import StructureBuilder
from GDPy.computation.utils import copy_minimal_frames

from GDPy.core.node import AbstractNode

#@registers.modifier.register
@registers.builder.register
class PerturbatorBuilder(StructureBuilder):

    name = "perturbater"

    """Perturb positions of input structures.

    TODO:
        1. Perturb cell.
        2. Perturb distances, angles...
        3. Check if perturbed structures are valid (too close distance).

    """

    def __init__(self, eps: float=0.2, directory="./", random_seed=1112, *args, **kwargs):
        """"""
        super().__init__(directory=directory, random_seed=random_seed)

        self.eps = eps # unit Ang

        return
    
    def run(self, substrates: List[Atoms], *args, **kwargs):
        """"""
        super().run(*args, **kwargs)

        frames, _ = copy_minimal_frames(substrates)
        for atoms in frames:
            natoms = len(atoms)
            pos_drift = self.rng.random((natoms,3))
            atoms.positions += pos_drift*self.eps

        return frames


if __name__ == "__main__":
    ...