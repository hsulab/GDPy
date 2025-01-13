#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple

import numpy as np
from ase import Atoms, units
from ase.ga.utilities import closest_distances_generator
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

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


def convert_composition_to_list(composition: dict, region) -> List[Tuple[Atoms, int]]:
    """"""
    # - define the composition of the atoms to optimise
    blocks = []
    for k, v in composition.items():
        k = convert_string_to_atoms(k)
        if isinstance(v, int):  # number
            v = v
        else:  # string command
            data = v.split()
            if data[0] == "density":
                v = compute_molecule_number_from_density(
                    np.sum(k.get_masses()), region.get_volume(), density=float(data[1])
                )
            else:
                raise RuntimeError(f"Unrecognised composition {k:v}.")
        blocks.append((k, v))

    # - check if there is any molecule
    for k, v in blocks:
        if len(k) > 1:
            use_tags = True
            break
    else:
        use_tags = False

    # atom_numbers = [] # atomic number of inserted atoms
    # for species, num in composition_blocks:
    #    numbers = []
    #    for s, n in ase.formula.Formula(species.get_chemical_formula()).count().items():
    #        numbers.extend([ase.data.atomic_numbers[s]]*n)
    #    atom_numbers.extend(numbers*num)

    return blocks


class remove_vacuum(Operation):

    cache: str = "cache_frames.xyz"

    def __init__(self, structures, thickness: float = 20.0, directory="./") -> None:
        """"""
        input_nodes = [structures]
        super().__init__(input_nodes, directory)

        self.thickness = thickness

        return

    def forward(self, structures) -> List[Atoms]:
        """Remove some vaccum of structures.

        Args:
            structures: List[Atoms] or AtomsNDArray.

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

    def forward(self, structures) -> List[Atoms]:
        """Remove some vaccum of structures.

        Args:
            structures: List[Atoms] or AtomsNDArray.

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
