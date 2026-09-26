#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional

import numpy as np

from ase import Atoms

from .builder import StructureModifier


class DeformModifier(StructureModifier):

    """Expand or compress bulk.
    """

    name: str = "deform"

    def __init__(self, ratio: Optional[list[float]]=None, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)
        self.ratio = np.arange(0.6, 1.2, 0.05) if ratio is None else ratio

        return

    def run(self, substrates=None, size: int=1, *args, **kwargs) -> list[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate)
            frames.extend(curr_frames)

        return frames

    def _irun(self, substrate: Atoms) -> list[Atoms]:
        """"""
        cell = copy.deepcopy(substrate.get_cell(complete=True))

        frames = []
        for i in self.ratio:
            atoms = copy.deepcopy(substrate)
            atoms.set_cell(cell*(i)**(1/3.), scale_atoms=True)
            frames.append(atoms)

        return frames


if __name__ == "__main__":
    ...
