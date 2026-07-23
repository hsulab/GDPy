#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def merge_results(host_frames: list[Atoms], corr_frames: list[Atoms]):
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


if __name__ == "__main__":
    ...
