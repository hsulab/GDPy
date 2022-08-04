#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

#import random
import numpy as np 

from pathlib import Path

from ase import units
from ase.io import read, write
from ase.build import make_supercell
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .calculator.dp import DP

from .md.nosehoover import NoseHoover


def pertubate_stucture(cwd, atoms, amplitude, num=100): 
    # draw seed from numpy random generator
    drng = np.random.default_rng()
    seed = drng.integers(sys.maxsize) % (2**32)
    print('seed ', seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    # random atom positions
    natoms = len(atoms)
    random_motions = amplitude*rng.standard_normal((num,natoms,3))

    pertubated_frames = []
    for cur_motions in random_motions:
        new_atoms = atoms.copy()
        new_atoms.positions += cur_motions
        pertubated_frames.append(new_atoms)
    
    write(cwd/'pframes.xyz', pertubated_frames)

    return 

def per_main():
    print('===== GDPy =====')
    test_dir = "/users/40247882/projects/oxides/dptrain"
    test_dir = Path(test_dir)

    init_stru = 'Pt3O4_opt.xyz' # initial structure path

    atoms = read(test_dir/init_stru)
    atoms = make_supercell(atoms, 2.0*np.eye(3))
    # print(atoms.cell)

    pertubate_stucture(test_dir, atoms, 0.1)

    pass

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
