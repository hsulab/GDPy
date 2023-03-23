#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from GDPy.core.operation import Operation

class perturb(Operation):

    """Perturb positions of input structures.
    """

    def __init__(self, frames, eps=0.1, rng=np.random):
        super().__init__([frames])

        self.eps = eps
        self.rng = rng

        return
    
    def forward(self, frames):
        """"""
        # TODO: if computed, use cached results
        new_frames = frames.copy()

        for atoms in new_frames:
            natoms = len(atoms)
            pos_drift = self.rng.random((natoms,3))
            atoms.positions += pos_drift*self.eps

        return new_frames


if __name__ == "__main__":
    ...