#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms

from GDPy.builder.builder import StructureGenerator


class DimerGenerator(StructureGenerator):

    def __init__(self, params, directory=Path.cwd(), *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.elements = params.get("elements", None)
        assert self.elements, "DimerGenerator needs elements as a param."
        assert len(self.elements) == 2, "DimerGenerator needs two chemical symbols as elements."

        self.number = params.get("number", 21)
        self.distance = params.get("distance", [0.8, 2.8])
        assert len(self.distance) == 2, "DimerGenerator needs min and max for the distance."

        return

    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        dmin, dmax = self.distance
        distances = np.linspace(dmin, dmax, self.number)

        frames = []
        for dis in distances:
            atoms = Atoms(
                symbols=self.elements, 
                positions=[
                    [0., 0., 0.],
                    [0., 0., dis]
                ],
                cell = 10.*np.eye(3),
                pbc=[True,True,True]
            )
            atoms.set_constraint(FixAtoms(indices=[0]))
            frames.append(atoms)

        return frames


if __name__ == "__main__":
    pass