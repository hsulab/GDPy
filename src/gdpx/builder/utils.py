#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase import Atoms, units
from ase.io import read, write

from gdpx.core.operation import Operation
from gdpx.data.array import AtomsNDArray


def compute_molecule_number_from_density(
    molecular_mass: float, volume: float, density: float
) -> int:
    """Compute the number of molecules in the region with a given density.

    Args:
        moleculer_mass: unit in g/mol.
        volume: unit in Ang^3.
        density: unit in g/cm^3.

    Returns:
        Number of molecules in the region.

    """
    number = (density / molecular_mass) * volume * units._Nav * 1e-24

    return int(number)


class remove_vacuum(Operation):

    cache: str = "cache_frames.xyz"

    def __init__(self, structures, thickness: float = 20.0, directory="./") -> None:
        """"""
        input_nodes = [structures]
        super().__init__(input_nodes, directory)

        self.thickness = thickness

        return

    def forward(self, structures) -> list[Atoms]:
        """Remove some vaccum of structures.

        Args:
            structures: A list of Atoms or AtomsNDArray.

        """
        super().forward()

        if isinstance(structures, AtomsNDArray):
            frames = structures.get_marked_structures()
        else:
            frames = structures

        # TODO: convert to atoms_array?
        cache_fpath = self.directory / self.cache
        if cache_fpath.exists():
            frames = read(cache_fpath, ":")
        else:
            for a in frames:
                a.cell[2, 2] -= self.thickness
            write(cache_fpath, frames)

        self.status = "finished"

        return frames


class reset_cell(Operation):

    cache: str = "cache_frames.xyz"

    def __init__(self, structures, cell, directory="./") -> None:
        """"""
        input_nodes = [structures]
        super().__init__(input_nodes, directory)

        self.cell = np.array(cell)

        return

    def forward(self, structures) -> list[Atoms]:
        """Remove some vaccum of structures.

        Args:
            structures: A list of Atoms or AtomsNDArray.

        """
        super().forward()

        if isinstance(structures, AtomsNDArray):
            frames = structures.get_marked_structures()
        else:
            frames = structures

        # TODO: convert to atoms_array?
        cache_fpath = self.directory / self.cache
        if cache_fpath.exists():
            frames = read(cache_fpath, ":")
        else:
            center_of_cell = np.sum(self.cell, axis=0) / 2.0
            for a in frames:
                com = a.get_center_of_mass()
                a.set_cell(self.cell)
                a.positions -= com - center_of_cell
            write(cache_fpath, frames)

        self.status = "finished"

        return frames


if __name__ == "__main__":
    ...
