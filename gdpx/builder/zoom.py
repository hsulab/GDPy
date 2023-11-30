#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from .builder import StructureModifier


class ZoomModifier(StructureModifier):

    """Extend or compress bulk.
    """

    def __init__(self, coefs: List[float]=None, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)
        if coefs is None:
            coefs = np.arange(0.6, 1.8, 0.05)
        self.coefs = coefs

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
        volume = substrate.get_volume()
        cell = copy.deepcopy(substrate.get_cell(complete=True))

        frames = []
        for i in self.coefs:
            atoms = copy.deepcopy(substrate)
            atoms.set_cell(cell*(i)**(1/3.), scale_atoms=True)
            frames.append(atoms)

        return frames
    

if __name__ == "__main__":
    ...