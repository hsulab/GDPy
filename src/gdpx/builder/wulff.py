#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np

from ase import Atoms
from ase.cluster import wulff_construction

from .builder import StructureBuilder


class WulffConstructionBuilder(StructureBuilder):

    name = "wulff_construction"

    def __init__(self, vacuum_size: float = -1, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        self.vacuum_size = vacuum_size

        self.parameters = dict(rounding="closest", latticeconstant=None)
        self.parameters.update(**kwargs)

        return

    def run(self, size: int = 1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs)

        frames = self._irun()

        return frames

    def _irun(self) -> List[Atoms]:
        """"""
        atoms = wulff_construction(
            symbol=self.parameters["symbol"],
            surfaces=self.parameters["surfaces"],
            energies=self.parameters["energies"],
            size=self.parameters["num_atoms"],
            structure=self.parameters["crystal_structure"],
            rounding=self.parameters["rounding"],
            latticeconstant=self.parameters["latticeconstant"],
        )

        if self.vacuum_size > 0.0:
            lengths = np.max(atoms.positions, axis=0) - np.min(atoms.positions, axis=0)
            lengths += self.vacuum_size
            cell = np.zeros((3, 3))
            np.fill_diagonal(cell, lengths)
            atoms.pbc = True
            atoms.cell = cell
            atoms.positions += np.sum(cell, axis=0)/2. - atoms.get_center_of_mass()

        return [atoms]


if __name__ == "__main__":
    ...
