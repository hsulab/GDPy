#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Optional, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell

from .builder import StructureModifier


class ScaleModifier(StructureModifier):

    """Make a supercell and scale its box.
    """

    def __init__(self, supercell: Optional[List[int]]=None, cubic: bool= False, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.supercell = np.array(supercell if supercell else [1,1,1])
        self.cubic = cubic

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
        new_atoms = make_supercell(substrate, np.diag(self.supercell))
        if self.cubic:  # TODO: This works only for an ortho box.
            new_cell = np.diag([np.max(new_atoms.cell.cellpar()[:3])]*3)
            new_atoms.set_cell(new_cell, scale_atoms=True)

        return [new_atoms]
    

if __name__ == "__main__":
    ...
