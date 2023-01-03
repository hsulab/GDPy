#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

import ase.optimize
from ase.constraints import FixAtoms
from ase.constraints import UnitCellFilter

from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.validator.validator import AbstractValidator

"""Validate minima and relative energies...
"""

def make_clean_atoms(atoms_, results=None):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc())
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms

class MinimaValidator(AbstractValidator):

    """Run minimisation on various configurations and compare relative energy.

    """

    def run(self):
        driver = self.pm.create_driver(self.task_params["driver"])
        global_constraint = driver._org_params["run"].get("constraint", None)

        params = self.task_params

        # - read structures
        names, frames, cons_info = [], [], []
        for name, stru_info in params["structures"].items():
            if isinstance(stru_info, dict):
                stru_path = stru_info.get("file", None)
                indices = stru_info.get("index", "-1")
                atoms = read(stru_path, indices)
                cons = stru_info.get("constraint", None)
            else:
                assert isinstance(stru_info, str), "Unsupported structure info."
                atoms = read(stru_info, "-1")
                cons = None
            if cons is None:
                cons = global_constraint
            names.append(name)
            frames.append(atoms)
            cons_info.append(cons)

        # - minimise structures
        new_frames = []
        for name, atoms_, cons in zip(names, frames, cons_info):
            self.logger.info(f"=== run minimisation for {name} ===")
            driver.directory = self.directory / name
            atoms = driver.run(atoms_, constraint=cons) 
            # - get clean results
            results = dict(
                energy = atoms.get_potential_energy(),
                forces = copy.deepcopy(atoms.get_forces())
            )
            atoms = make_clean_atoms(atoms, results)
            new_frames.append(atoms)

        # - process results and output
        ref_energies, new_energies = [], []
        ref_maxforces, new_maxforces = [], []
        displacements = []
        for name, ref_atoms, new_atoms in zip(names, frames, new_frames):
            # -- ene
            ref_energies.append(ref_atoms.get_potential_energy())
            new_energies.append(new_atoms.get_potential_energy())
            # -- frc
            ref_maxforces.append(np.max(np.fabs(ref_atoms.get_forces(apply_constraint=True))))
            new_maxforces.append(np.max(np.fabs(new_atoms.get_forces(apply_constraint=True))))
            # -- dis
            # TODO: not for bulk systems
            vector = new_atoms.get_positions() - ref_atoms.get_positions()
            vmin, vlen = find_mic(vector, new_atoms.get_cell())
            displacements.append(np.linalg.norm(vlen))
        
        #data = np.array([names, ref_energies, new_energies, ref_maxforces, new_maxforces, displacements]).T
        #np.savetxt(
        #    self.directory/"abs.dat", data,
        #    fmt="%24s  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f",
        #    #header=("{:<24s}  "+"{:<12.4f}  "*5).format("Name", "RefEne", "NewEne", "RefMaxF", "NewMaxF", "Drmse")
        #)

        content = ("{:<24s}  "+"{:<12s}  "*5+"\n").format("Name", "RefEne", "NewEne", "RefMaxF", "NewMaxF", "Drmse")
        for name, ref_ene, new_ene, ref_maxfrc, new_maxfrc, dis in zip(
            names, ref_energies, new_energies, ref_maxforces, new_maxforces, displacements
        ):
            content += ("{:>24s}  "+"{:>12.4f}  "*5+"\n").format(
                name, ref_ene, new_ene, ref_maxfrc, new_maxfrc, dis
            )

        with open(self.directory/"abs.dat", "w") as fopen:
            fopen.write(content)
        self.logger.info(content)

        # - parse relative energies
        ene_dict = {}
        for name, ref_ene, new_ene in zip(names, ref_energies, new_energies):
            ene_dict[name] = [ref_ene, new_ene]
            ...
        equations = params.get("equations", None)
        if equations is not None:
            ref_rets, new_rets = [], []
            for eqn in equations:
                ref_eqn_data, new_eqn_data = [], []
                data = eqn.strip().split()
                for e in data:
                    if (not e.isdigit()) and (e not in ["+", "-", "*", "/"]):
                        if e in names:
                            ref_eqn_data.append(
                                str(round(float(ene_dict[e][0]),4))
                            )
                            new_eqn_data.append(
                                str(round(float(ene_dict[e][1]),4))
                            )
                    else:
                        ref_eqn_data.append(e)
                        new_eqn_data.append(e)
                #print(ref_eqn_data)
                #print(new_eqn_data)
                ref_eqn = "".join(ref_eqn_data)
                new_eqn = "".join(new_eqn_data)
                ref_rets.append(eval(ref_eqn))
                new_rets.append(eval(new_eqn))
            
            self.logger.info("=== Chemical Equations ===")
            content = "{:<12s}  {:<12s}  {:<40s}\n".format("Ref", "New", "Eqn")
            for eqn, ref_val, new_val in zip(equations, ref_rets, new_rets):
                content += "{:>12.4f}  {:>12.4f}  {:>40s}\n".format(ref_val, new_val, eqn)
            self.logger.info(content)

        return


if __name__ == "__main__":
    pass