#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional, List

import numpy as np

from ase import Atoms

from ..builder import StructureModifier
from .. import str2array
from .intercoord import compute_bond_angles, compute_angle_jacobian, optimisation_step


class ScanAngleModifier(StructureModifier):

    name = "scan_angle"

    #: Maximum number of updates of positions.
    MAX_ATTEMPTS_UPDATE: int = 100

    #: Tolerance of target internal coordinates.
    TOL_INTCOORD: float = 1e-4

    def __init__(self, angle: List[int], target: str, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.angle = np.array(angle).reshape(-1, 3)
        if isinstance(target, str):
            self.target = str2array(target)
        else: # assume it is a plain list
            self.target = np.array(target)

        return

    def run(self, substrates: Optional[List[Atoms]], size: int = 1, *args, **kwargs):
        """"""
        super().run(substrates=substrates, *args, *kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames

    def _irun(self, substrate: Atoms, size=1, *args, **kwargs) -> List[Atoms]:
        """"""
        targets = self.target[:, np.newaxis]
        trimers = self.angle

        frames = []
        for target in targets:
            curr_atoms = copy.deepcopy(substrate)
            curr_atoms = self._approx_structure(curr_atoms, target, trimers)
            # wrap structures as sometimes the atoms are out of box
            curr_atoms.wrap()
            frames.append(curr_atoms)

        return frames

    def _approx_structure(self, atoms: Atoms, targets, trimers):
        """"""
        natoms = len(atoms)

        maxstep = 1.00

        # - compute pseudo inverse of the jacobian matrix
        self._debug(f"{targets =}")
        positions = copy.deepcopy(atoms.positions)
        for _ in range(self.MAX_ATTEMPTS_UPDATE):
            internals = compute_bond_angles(positions, trimers)
            self._debug(f"internals: {internals}")
            disp = targets - internals
            if np.max(np.fabs(disp)) < self.TOL_INTCOORD:
                break
            ang_jac_ = compute_angle_jacobian(positions, trimers)
            jac = ang_jac_.reshape(-1, natoms * 3)
            positions = copy.deepcopy(positions) + maxstep * optimisation_step(
                jac, disp
            )
        else:
            # warnings.warn("Iterative approximation is not converged.", UserWarning)
            self._print("Iterative approximation is not converged.")

        # - update positions
        atoms.positions = positions

        return atoms


if __name__ == "__main__":
    ...
