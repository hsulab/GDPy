#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Optional, List

import numpy as np

from ase import Atoms

from .builder import StructureModifier 
from .utils import str2list_int, check_overlap_neighbour


class PerturbatorBuilder(StructureModifier):

    name = "perturbater"

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    #: Number of attempts to create a number of candidates.
    #       if 10 structures are to create, run will try 5*10=50 times.
    MAX_TIMES_SIZE: int = 5

    """Perturb positions of input structures.

    TODO:
        1. Perturb distances, angles...

    """

    def __init__(
            self, eps: Optional[float]=None, ceps: Optional[float]=None, 
            group: Optional[str]=None,
            covalent_ratio=[0.8, 2.0],
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

        # - apply perturbation on a group of atoms, default is all
        self.group = group
        if self.group is not None:
            self.group = str2list_int(self.group, convention="lmp")

        # - distance check
        self.covalent_ratio = covalent_ratio
        self.MAX_TIMES_SIZE = max_times_size

        return
    
    def run(self, substrates: Optional[List[Atoms]]=None, size:int=1, *args, **kwargs) -> List[Atoms]:
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
                    if self.group is not None:
                        pos_drift_ = np.zeros((natoms, 3))
                        pos_drift_[self.group] = pos_drift[self.group]
                        pos_drift = pos_drift_
                    atoms.positions += pos_drift*self.eps
                if all(atoms.pbc) and self.ceps is not None:
                    lat_drift = self.rng.random((3, 3))
                    atoms.cell += lat_drift*self.ceps
                if check_overlap_neighbour(atoms, self.covalent_ratio):
                    frames.append(atoms)
            else:
                break
        else:
            nframes = len(frames)
            raise RuntimeError(
                f"Failed to create {size} structures, only {nframes} are created."
            )

        return frames


if __name__ == "__main__":
    ...
