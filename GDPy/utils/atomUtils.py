#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import argparse
from pathlib import Path

import numpy as np

from ase.io import read, write


""" this temporarily stores some utilities
    for atoms object
"""

def check_convergence(atoms, fmax=0.05):
    """Check the convergence of the trajectory"""

    forces = atoms.get_forces()

    max_force = np.max(np.fabs(forces))

    converged = False
    if max_force < fmax:
        converged = True 

    return converged

def merge_xyz():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pattern",
        help="pattern to find xyz files"
    )
    args = parser.parse_args()

    #pattern = "s2hcp"
    pattern = args.pattern
    cwd = Path.cwd()

    sorted_paths = []
    for p in cwd.glob(pattern+"*.xyz"):
        sorted_paths.append(p)
    sorted_paths.sort()

    frames = []
    for p in sorted_paths:
        cur_frames = read(p, ":")
        print(p.stem)
        for atoms in cur_frames:
            atoms.info["source"] = str(p.stem)
        print(p, " #frames: ", len(cur_frames))
        frames.extend(cur_frames)
    print("TOTAL #frames: ", len(frames))
    write(pattern+"-0906.xyz", frames)

    for p in sorted_paths:
        shutil.move(p, cwd / ("bak."+p.name))


def sort_atoms(atoms):
    # sort atoms by symbols and z-positions especially for supercells 
    numbers = atoms.numbers 
    zposes = atoms.positions[:,2].tolist()
    sorted_indices = np.lexsort((zposes,numbers))
    atoms = atoms[sorted_indices]

    return atoms


def try_sort():
    frames = read('./test_data.xyz', ':')
    
    new_frames = [sort_atoms(atoms) for atoms in frames]
    write('new_test.xyz', new_frames)

if __name__ == '__main__':
    pass
