#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write
from ase.build import make_supercell

def sort_atoms(atoms):
    # sort atoms by symbols and z-positions especially for supercells 
    numbers = atoms.numbers 
    zposes = atoms.positions[:,2].tolist()
    sorted_indices = np.lexsort((zposes,numbers))
    atoms = atoms[sorted_indices]

    return atoms

init_stru = './opts/Pt3O4_opt.xyz'
atoms = read(init_stru)
atoms = make_supercell(atoms, 2.0*np.eye(3)) # (2x2x2) cell
atoms = sort_atoms(atoms)
print(atoms.cell)

write('Pt32.data', atoms, format='lammps-data')

if __name__ == '__main__':
    pass
