#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

from ase import Atoms
from ase.io import read, write

from ..core.operation import Operation
from ..data.array import AtomsNDArray

"""Some extra operations.
"""


class remove_vacuum(Operation):

    cache: str = "cache_frames.xyz"

    def __init__(self, structures, thickness: float=20., directory="./") -> None:
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
        cache_fpath = self.directory/self.cache
        if cache_fpath.exists():
            frames = read(cache_fpath, ":")
        else:
            for a in frames:
                a.cell[2, 2] -= self.thickness
            write(cache_fpath, frames)
        
        self.status = "finished"

        return frames


if __name__ == "__main__":
    ...