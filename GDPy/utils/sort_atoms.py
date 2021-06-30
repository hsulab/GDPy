#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write


def sort_atoms(atoms):
    # sort atoms by symbols and z-positions especially for supercells 
    numbers = atoms.numbers 
    zposes = atoms.positions[:,2].tolist()
    sorted_indices = np.lexsort((zposes,numbers))
    atoms = atoms[sorted_indices]

    return atoms


frames = read('./test_data.xyz', ':')

new_frames = [sort_atoms(atoms) for atoms in frames]
write('new_test.xyz', new_frames)

if __name__ == '__main__':
    pass
