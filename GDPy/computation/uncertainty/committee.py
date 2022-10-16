#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import List

import numpy as np

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError


class CommitteeUncertaintyEstimator():

    """Use model committee to estimate prediction uncertainty.
    """

    #: Working directory.
    _directory = "./"

    def __init__(self, calculators, properties=None) -> None:
        """Require a list of calculators and target properties."""
        self.committee = calculators
        self._directory = pathlib.Path(self._directory)

        return
    
    @property
    def directory(self):
        """"""

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_).resolve()

        return 

    def estimate(self, frames: List[Atoms]) -> List[Atoms]:
        """Use committee to estimate uncertainty
        """
        for i, c in enumerate(self.committee):
            c.directory = str(self.directory/f"c{i}")

        # max_devi_e, max_devi_f
        # TODO: directory where estimate?
        for atoms in frames:
            cmt_tot_energy, cmt_energies, cmt_forces = [], [], []
            for c in self.committee:
                c.reset()
                atoms.calc = c
                # - total energy
                energy = atoms.get_potential_energy()
                cmt_tot_energy.append(energy)
                # - atomic energies
                try:
                    energies = atoms.get_potential_energies()
                except PropertyNotImplementedError:
                    energies = [1e8]*len(atoms)
                cmt_energies.append(energies)
                # - atomic forces
                forces = atoms.get_forces()
                cmt_forces.append(forces)
            
            # --- total energy
            cmt_tot_energy = np.array(cmt_tot_energy)
            tot_energy_devi = np.sqrt(np.var(cmt_tot_energy))
            atoms.info["te_devi"] = tot_energy_devi

            # --- atomic energies
            cmt_energies = np.array(cmt_energies)
            ae_devi = np.sqrt(np.var(cmt_energies, axis=0))
            atoms.arrays["ae_devi"] = ae_devi
            atoms.info["max_devi_ae"] = np.max(ae_devi)

            # --- atomic forces
            cmt_forces = np.array(cmt_forces)
            force_devi = np.sqrt(np.var(cmt_forces, axis=0))
            atoms.arrays["force_devi"] = force_devi
            atoms.info["max_devi_f"] = np.max(force_devi)

        return frames


if __name__ == "__main__":
    pass