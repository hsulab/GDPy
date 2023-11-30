#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms

from .builder import StructureBuilder


class DimerBuilder(StructureBuilder):

    def __init__(self, elements: List[str], distances: List[float]=[0.8,2.5,0.05], directory=Path.cwd(), *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.elements = elements
        assert len(self.elements) == 2, "DimerBuilder needs two chemical symbols as elements."

        self.distances = distances
        assert len(self.distances) == 3, "DimerBuilder needs min, max and intv for the distance."

        return

    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs)

        dmin, dmax, intv = self.distances
        distances = np.arange(dmin, dmax+intv, intv)

        frames = []
        for dis in distances:
            atoms = Atoms(
                symbols=self.elements, 
                positions=[
                    [0., 0., 0.],
                    [0., 0., dis]
                ],
                #cell = 20.*np.eye(3),
                cell = [[19., 0., 0.], [0., 20., 0.], [0., 0., 21.]],
                pbc=[True,True,True]
            )
            atoms.set_constraint(FixAtoms(indices=[0]))
            frames.append(atoms)

        return frames


if __name__ == "__main__":
    ...
