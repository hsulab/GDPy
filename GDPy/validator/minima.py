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

from GDPy.core.register import registers
from GDPy.validator.validator import AbstractValidator
from GDPy.computation.worker.drive import DriverBasedWorker
from GDPy.builder.constraints import set_constraint

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

@registers.validator.register
class MinimaValidator(AbstractValidator):

    """Run minimisation on various configurations and compare relative energy.

    TODO: 

        Support the comparison of minimisation trajectories.

    """

    def run(self, dataset: dict, worker: DriverBasedWorker, *args, **kwargs):
        """"""
        task_params = copy.deepcopy(self.task_params)

        # TODO: assume dataset is a dict of frames
        pre_dataset = {}
        for k, curr_frames in dataset.items():
            nframes = len(curr_frames)

            cached_pred_fpath = self.directory/k/ "pred.xyz"
            if not cached_pred_fpath.exists():
                worker.directory = self.directory/k
                worker.batchsize = nframes

                #worker._share_wdir = True

                worker.run(curr_frames)
                worker.inspect(resubmit=True)
                if worker.get_number_of_running_jobs() == 0:
                    pred_frames = worker.retrieve(
                        ignore_retrieved=False,
                    )
                else:
                    # TODO: ...
                    ...
                write(cached_pred_fpath, pred_frames)
            else:
                pred_frames = read(cached_pred_fpath, ":")
            
            pre_dataset[k] = pred_frames
        
        # -
        key = "composites"
        
        # - restore constraints
        cons_text = worker.driver.as_dict()["constraint"]
        for ref_atoms, pre_atoms in zip(dataset[key], pre_dataset[key]):
            set_constraint(ref_atoms, cons_text, ignore_attached_constraints=True)
            set_constraint(pre_atoms, cons_text, ignore_attached_constraints=True)
        
        # - compare ...
        ref_energies = np.array([a.get_potential_energy() for a in dataset[key]])
        pre_energies = np.array([a.get_potential_energy() for a in pre_dataset[key]])

        ref_maxforces = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in dataset[key]])
        pre_maxforces = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in pre_dataset[key]])

        # - compute shifts if any
        ref_ene_shift = np.sum([x*y for x, y in task_params["ref_ene_shift"]])
        pre_ene_shift = np.sum([x*pre_dataset[y][0].get_potential_energy() for x, y in task_params["pre_ene_shift"]])

        ref_rel_energies = ref_energies - ref_ene_shift
        pre_rel_energies = pre_energies - pre_ene_shift

        # -- disp
        disps = [] # displacements
        for ref_atoms, pre_atoms in zip(dataset[key], pre_dataset[key]):
            vector = pre_atoms.get_positions() - ref_atoms.get_positions()
            vmin, vlen = find_mic(vector, pre_atoms.get_cell())
            disps.append(np.linalg.norm(vlen))

        content = ("{:<24s}  "+"{:<12s}  "*7+"\n").format("Name", "RefEne", "NewEne", "RefMaxF", "NewMaxF", "Drmse", "RefRelEne", "NewRelEne")
        for i, (ref_ene, new_ene, ref_maxfrc, new_maxfrc, dis, ref_rel_ene, pre_rel_ene) in enumerate(
            zip(ref_energies, pre_energies, ref_maxforces, pre_maxforces, disps, ref_rel_energies, pre_rel_energies)
        ):
            content += ("{:<24s}  "+"{:<12.4f}  "*7+"\n").format(
                str(i), ref_ene, new_ene, ref_maxfrc, new_maxfrc, dis, ref_rel_ene, pre_rel_ene
            )

        with open(self.directory/"abs.dat", "w") as fopen:
            fopen.write(content)
        
        self._print(content)

        return

    def _compute_chemical_equations(self, dataset, worker, *args, **kwargs):
        """"""
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