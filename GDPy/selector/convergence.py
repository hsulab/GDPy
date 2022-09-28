#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class ConvergenceSelector(AbstractSelector):

    """ find geometrically converged frames
    """

    name = "convergence"

    default_parameters = dict(
        fmax = 0.05 # eV/AA
    )

    def __init__(self, directory=Path.cwd(), *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """"""
        # NOTE: input atoms should have constraints attached
        selected_indices = []
        for i, atoms in enumerate(frames):
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            if maxforce < self.fmax:
                selected_indices.append(i)
        
        # - output
        data = []
        for s in selected_indices:
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            ae = atoms.get_potential_energy() / natoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            data.append([s, confid, natoms, ae, maxforce])
        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %12.4f  %12.4f",
                #fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce".split()
                )
            )

        return selected_indices


if __name__ == "__main__":
    pass