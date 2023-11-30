#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import List

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from ..core.operation import Operation


def merge_results(host_frames: List[Atoms], corr_frames: List[Atoms]):
    """"""
    frames = []
    for host, corr in zip(host_frames, corr_frames):
        # TODO: check atoms consistent?
        atoms = Atoms(
            symbols=copy.deepcopy(host.get_chemical_symbols()),
            positions=copy.deepcopy(host.get_positions()),
            cell=copy.deepcopy(host.get_cell(complete=True)),
            pbc=copy.deepcopy(host.get_pbc()),
            tags = host.get_tags() # retain this for molecules
        )
        if host.get_kinetic_energy() > 0.: # retain this for MD
            atoms.set_momenta(host.get_momenta()) 
        # TODO: add info and arrays?
        calc = SinglePointCalculator(
            atoms, 
            energy = host.get_potential_energy() + corr.get_potential_energy(),
            forces = host.get_forces(apply_constraint=True) + corr.get_forces(apply_constraint=True),
        )
        atoms.calc = calc
        frames.append(atoms)

    return frames


class correct(Operation):

    """"""

    def __init__(self, structures, computer, directory="./") -> None:
        """"""
        input_nodes = [structures, computer]
        super().__init__(input_nodes, directory)

        return
    
    def forward(self, structures, computer):
        """"""
        super().forward()

        computer = computer[0] # TODO: accept several computers...
        computer._share_wdir = True

        # - structures should be a Tempdata node for now
        computer_status = [False for i in range(len(structures))]
        for i, (curr_name, curr_frames) in enumerate(structures):
            nframes = len(curr_frames)
            print(curr_name, len(curr_frames))
            computer.directory = self.directory/curr_name
            computer.batchsize = nframes
            computer.run(curr_frames)
            computer.inspect(resubmit=True)
            if (computer.get_number_of_running_jobs() == 0):
                computer_status[i] = True
                continue
        
        if all(computer_status):
            self._print("correction computation finished...")
            self.status = "finished"
            new_structures = []
            for i, (curr_name, curr_frames) in enumerate(structures):
                computer.directory = self.directory/curr_name
                # TODO: neew a better unified interface
                curr_corr_frames = computer.retrieve()
                if not computer._share_wdir:
                    curr_corr_frames = itertools.chain(*curr_corr_frames)
                curr_new_frames = merge_results(curr_frames, curr_corr_frames)
                write(self.directory/f"{curr_name}"/"merged.xyz", curr_new_frames)
                new_structures.append([curr_name, curr_new_frames])

        return new_structures


if __name__ == "__main__":
    ...