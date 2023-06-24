#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np 

from ase import Atoms
from ase.io import read, write 

from .builder import StructureModifier

"""Cleave surface from the bulk structure.
Surface miller index <= 2 
"""


class CleaveSurfaceModifier(StructureModifier):

    name: str = "cleave_surface"

    def __init__(
            self, backend="ase", max_miller: int=2, slab_size: float=4.0, vaccum_size: float=12.0, 
            center_slab=False, normal=1, substrates=None, *args, **kwargs
        ):
        """"""
        super().__init__(substrates, *args, **kwargs)

        self.backend = backend

        self.max_miller = max_miller
        self.slab_size = slab_size
        self.vaccum_size = vaccum_size
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
                    min_vacuum_size = self.vaccum_size, 
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