#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

import ase.optimize
from ase.constraints import FixAtoms
from ase.constraints import UnitCellFilter

from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.validator.validator import AbstractValidator
from gdpx.worker.drive import DriverBasedWorker
from gdpx.builder.constraints import set_constraint
from ..data.array import AtomsNDArray

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

    TODO: 

        Support the comparison of minimisation trajectories.

    """

    def __init__(self, ene_shift=[], *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.ene_shift = ene_shift

        return
    
    def _process_data(self, data) -> List[Atoms]:
        """"""
        data = AtomsNDArray(data)

        # We need a List of Atoms (ndim=1).
        if data.ndim == 1:
            data = data.tolist()
        #elif data.ndim == 2: # assume it is from extract_cache...
        #    data = data.tolist()
        #elif data.ndim == 3: # assume it is from a compute node...
        #    data_ = []
        #    for d in data[:]: # TODO: add squeeze method?
        #        data_.extend(d)
        #    data = data
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data

    def run(self, dataset: dict, worker: DriverBasedWorker, *args, **kwargs):
        """"""
        # TODO: assume dataset is a dict of frames
        pre_dataset = {}
        for k, curr_data in dataset.items():
            curr_frames = self._process_data(curr_data)
            nframes = len(curr_frames)

            cached_pred_fpath = self.directory/k/ "pred.xyz"
            if not cached_pred_fpath.exists():
                worker.directory = self.directory/k
                worker.batchsize = nframes

                worker.run(curr_frames)
                worker.inspect(resubmit=True)
                if worker.get_number_of_running_jobs() == 0:
                    # -- get end frames
                    trajectories = worker.retrieve(
                        include_retrieved=True,
                    )
                    pred_frames = [t[-1] for t in trajectories]
                else:
                    # TODO: ...
                    ...
                write(cached_pred_fpath, pred_frames)
            else:
                pred_frames = read(cached_pred_fpath, ":")
            
            pre_dataset[k] = pred_frames
        
        # -
        keys = list(dataset.keys())
        for key in keys:
            self._print(f"group {key}")
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
            if key == "composites":
                if self.ene_shift:
                    ref_ene_shift = np.sum([x*dataset[y][0].get_potential_energy() for x, y in self.ene_shift])
                    pre_ene_shift = np.sum([x*pre_dataset[y][0].get_potential_energy() for x, y in self.ene_shift])
            else:
                ref_ene_shift = 0.
                pre_ene_shift = 0.

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

            with open(self.directory/f"abs-{key}.dat", "w") as fopen:
                fopen.write(content)
        
            self._print(content)

        return


if __name__ == "__main__":
    ...