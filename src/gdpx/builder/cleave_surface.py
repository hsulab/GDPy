#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np 

from ase import Atoms
from ase.io import read, write 

from .builder import StructureModifier


def get_rotation_matrix_from_string(orientation: str):
    """"""
    vec_map = {
        "+x": [+1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "+y": [0.0, +1.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, +1.0],
        "-z": [0.0, 0.0, -1.0],
    }

    axis_map = {
        "+x": ("+y", "+z", "+x")
    }

    rotation_matrix = np.array(
        [vec_map[o] for o in axis_map[orientation]]
    )

    return rotation_matrix


class AddVacuumModifier(StructureModifier):

    name: str = "add_vacuum"

    def __init__(self, orientation: str, vacuum_size: float=12.0, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        if orientation not in ["+x", "-x", "+y", "-y", "+z", "-z"]:
            raise RuntimeError(f"Improper orientation `{orientation}`.")
        self.orientation = orientation

        self.vacuum_size = vacuum_size

        return

    def run(self, substrates=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames

    def _irun(self, substrate: Atoms, size: int, *args, **kwargs) -> List[Atoms]:
        """"""
        new_atoms = copy.deepcopy(substrate)

        cell = new_atoms.get_cell(complete=True)
        cell = cell.array
        is_diagonal = np.all(cell == np.diag(np.diagonal(cell)))
        if not is_diagonal:
            raise RuntimeError(f"The substrate `{new_atoms}` has no diagonal cell.")

        # define rotation matrix to align x-axis with z-axis
        # rotation_matrix = np.array([
        #     [0, 1, 0],  # y -> x
        #     [0, 0, 1],  # z -> y
        #     [1, 0, 0],  # x -> z
        # ])
        rotation_matrix = get_rotation_matrix_from_string(self.orientation)
        
        # apply rotation
        positions = new_atoms.get_positions()
        rotated_positions = np.dot(positions, rotation_matrix.T)
        new_atoms.set_positions(rotated_positions)

        # apply rotation to the cell
        cell = new_atoms.get_cell(complete=True)
        rotated_cell = np.diag(np.dot(np.diagonal(cell), rotation_matrix.T))
        rotated_cell[2, 2] += self.vacuum_size

        new_atoms.set_cell(rotated_cell)

        return [new_atoms]


class CleaveSurfaceModifier(StructureModifier):

    """Cleave surface from the bulk structure.
    
    Currently, we only support surfaces iwth miller indices smaller
    than 2.

    """

    name: str = "cleave_surface"


    def __init__(
            self, backend="ase", max_miller: int=2, slab_size: float=4.0, vacuum_size: float=12.0, 
            center_slab=False, normal=1, substrates=None, *args, **kwargs
        ):
        """"""
        super().__init__(substrates, *args, **kwargs)

        self.backend = backend

        self.max_miller = max_miller
        self.slab_size = slab_size
        self.vacuum_size = vacuum_size
        self.center_slab = center_slab
        self.normal = normal

        return
    
    def run(self, substrates=None, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        if self.backend == "ase":
            from ase.lattice.cubic import FaceCenteredCubic
            #atoms = FaceCenteredCubic(
            #    symbol="Cu", 
            #    #directions=[[1,-1,0], [1,1,-2], [1,1,1]],
            #    directions=[[1,0,0], [1,1,-2], [1,1,1]],
            #    debug=2
            #)
            raise NotImplementedError()
        elif self.backend == "matgen":
            try:
                import pymatgen as mg
                import pymatgen.core.surface as mg_surface 
                from pymatgen.io.ase import AseAtomsAdaptor
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            except Exception as e:
                raise RuntimeError(e)
            for substrate in self.substrates:
                structure = AseAtomsAdaptor.get_structure(substrate)
                structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
                print(structure.lattice.d_hkl((1,1,1)))
                possible_slabs = mg_surface.generate_all_slabs(
                    structure, 
                    max_index = self.max_miller, 
                    min_slab_size = self.slab_size, 
                    min_vacuum_size = self.vacuum_size, 
                    center_slab = self.center_slab,
                    max_normal_search = self.normal,
                    in_unit_planes=True
                )
                frames.extend([AseAtomsAdaptor.get_atoms(s) for s in possible_slabs])
        else:
            ...

        return frames
    
    def _irun(self):
        """"""

        return


if __name__ == "__main__":
    ...
