#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from ase import Atoms

def get_properties(frames: List[Atoms], other_props = []):
    """Get properties of frames for comparison.

    Currently, only total energy and forces are considered.

    Returns:
        tot_symbols: shape (nframes,)
        tot_energies: shape (nframes,)
        tot_forces: shape (nframes,3)

    """
    tot_symbols, tot_energies, tot_forces = [], [], []

    for atoms in frames: # free energy per atom
        # -- basic info
        symbols = atoms.get_chemical_symbols()
        tot_symbols.extend(symbols)

        # -- energy
        energy = atoms.get_potential_energy() 
        tot_energies.append(energy)

        # -- force
        forces = atoms.get_forces(apply_constraint=False)
        tot_forces.extend(forces.tolist())

    return tot_symbols, tot_energies, tot_forces


if __name__ == "__main__":
    ...