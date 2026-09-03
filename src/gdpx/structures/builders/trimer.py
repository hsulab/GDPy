#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from .builder import StructureBuilder


class TrimerBuilder(StructureBuilder):

    name: str = "trimer"

    def __init__(
        self,
        elements: List[str],
        angles: List[float] = [0.8, 2.5, 0.05],
        distance: float=1.2,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.elements = elements
        if not (len(self.elements) == 3):
            raise RuntimeError("TrimerBuilder needs three chemical symbols and the first is the vertex.")

        self.angles = angles
        if not (len(self.angles) == 3):
            raise RuntimeError("TrimerBuilder needs min, max and intv for the angle in [rad].")

        self.distance = distance

        return

    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs)

        amin, amax, intv = self.angles
        angles = np.arange(amin, amax + intv, intv)
        self._print(f"{angles =}")

        frames = []
        for ang in angles:
            atoms = Atoms(
                symbols=self.elements,
                positions=[
                    [10.0, 10.0, 10.0], 
                    [10.0, 10.0+self.distance, 10.0],
                    [10.0, 10.0+self.distance*np.cos(ang), 10.0+self.distance*np.sin(ang)],
                ],
                cell=[[19.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 21.0]],
                pbc=[True, True, True],
            )
            atoms.set_constraint(FixAtoms(indices=[0]))
            frames.append(atoms)

        return frames


if __name__ == "__main__":
    ...
