#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np

from ase.io import read, write

from GDPy.validator.validator import AbstractValidator


class DimerValidator(AbstractValidator):

    def run(self):
        """"""
        params = self.task_params
        stru_fpaths = params["references"]

        for fpath in stru_fpaths:
            frames = read(fpath, ":")
            self._irun(frames)

        return

    def _irun(self, frames):
        """"""
        # - get references and calc mlp
        ref_distances, ref_energies = [], []
        mlp_distances, mlp_energies = [], []
        for atoms_ in frames:
            # - ref
            atoms = copy.deepcopy(atoms_)
            natoms = len(atoms)
            assert natoms == 2, "Input structure must be a dimer."
            positions = atoms.get_positions()
            ref_distances.append(
                atoms.get_distance(0, 1, mic=True, vector=False)
            )
            ref_energies.append(
                atoms.get_potential_energy()
            )
            # - mlp
            atoms.calc = self.pm.calc
            mlp_energies.append(
                atoms.get_potential_energy()
            )
        
        # - save data
        abs_errors = [x-y for x,y in zip(mlp_energies,ref_energies)]
        rel_errors = [(x/y)*100. for x,y in zip(abs_errors,ref_energies)]
        data = np.array([ref_distances,ref_energies,mlp_energies,abs_errors,rel_errors]).T

        fname = str(frames[0].get_chemical_formula())
        np.savetxt(
            self.directory/f"{fname}.dat", data, 
            fmt="%8.4f  %12.4f  %12.4f  %12.4f  %8.4f", 
            header="{:<8s}  {:<12s}  {:<12s}  {:<12s}  {:<8s}".format(
                "dis", "ref", "mlp", "abs", "rel [%]"
            )
        )

        return
    

if __name__ == "__main__":
    pass