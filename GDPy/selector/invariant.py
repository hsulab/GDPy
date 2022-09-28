#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List
from pathlib import Path

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class InvariantSelector(AbstractSelector):

    name = "invariant"

    default_parameters = dict()

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """"""
        selected_indices = list(range(len(frames)))

        # - output
        data = []
        for s in selected_indices:
            atoms = frames[s]
            # - add info
            selection = atoms.info.get("selection","")
            atoms.info["selection"] = selection+f"->{self.name}"
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