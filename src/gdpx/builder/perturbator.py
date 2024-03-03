#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from .builder import StructureModifier 
from .utils import check_overlap_neighbour


class PerturbatorBuilder(StructureModifier):

    name = "perturbater"

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    #: Number of attempts to create a number of candidates.
    #       if 10 structures are to create, run will try 5*10=50 times.
    MAX_TIMES_SIZE: int = 5

    """Perturb positions of input structures.

    TODO:
        1. Perturb cell.
        2. Perturb distances, angles...
        3. Check if perturbed structures are valid (too close distance).

    """

    def __init__(
            self, eps: float=None, ceps: float=None, covalent_ratio=[0.8, 2.0],
            max_times_size: int = 5, *args, **kwargs
        ):
        """Initialise a perturbator.

        Args:
            eps: Drift on atomic positions. Unit Ang.
            ceps: Drift on lattice constants. Unit Ang.

        """
        super().__init__(*args, **kwargs)

        # - check inputs
        if eps is None and ceps is None:
            raise RuntimeError("Either eps or ceps should be provided.")
        elif eps is not None and ceps is None:
            ...
        elif eps is None and ceps is not None:
            ...
        else:
            ...

        self.eps = eps
        self.ceps = ceps

        # - distance check
        self.covalent_ratio = covalent_ratio
        self.MAX_TIMES_SIZE = max_times_size

        return
    
    def run(self, substrates: List[Atoms]=None, size:int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate, size)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int):
        """"""
        frames = []
        for i in range(size*self.MAX_TIMES_SIZE):
            nframes = len(frames)
            if nframes < size:
                atoms = copy.deepcopy(substrate)
                if self.eps is not None:
                    natoms = len(atoms)
                    pos_drift = self.rng.random((natoms, 3))
                    atoms.positions += pos_drift*self.eps
                if all(atoms.pbc) and self.ceps is not None:
                    lat_drift = self.rng.random((3, 3))
                    atoms.cell += lat_drift*self.ceps
                if check_overlap_neighbour(atoms, self.covalent_ratio):
                    frames.append(atoms)
            else:
                break
        else:
            raise RuntimeError(
                f"Failed to create {size} structures, only {nframes} are created."
            )

        return frames


if __name__ == "__main__":
    ...