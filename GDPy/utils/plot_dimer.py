#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot dimer curve 
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt

from ase.io import read, write 

def parse_dimer_frames(xyzfile):
    frames = read(xyzfile, ':16') 
    dimer_symbols = frames[0].get_chemical_symbols()

    data = []
    for atoms in frames:
        assert len(atoms) == 2 
        energy = atoms.get_potential_energy() 
        dist = np.linalg.norm(atoms[0].position-atoms[1].position)
        data.append([dist,energy])
    data = np.array(data) 

    return dimer_symbols, data 

def parse_dimer_from_files(prefix):
    distances, energies = [], []
    for p in Path.cwd().glob(prefix):
        vasprun = p / 'vasprun.xml'
        atoms = read(vasprun, format='vasp-xml')
        distances.append(np.linalg.norm(atoms[0].position-atoms[1].position))
        energies.append(atoms.get_potential_energy())

    return distances, energies

def harvest_dimer_from_files(prefix):
    dimer_dirs = []
    for p in Path.cwd().glob(prefix):
        dimer_dirs.append(str(p))

    dimer_dirs.sort()

    dimer_frames = []
    for p in dimer_dirs:
        p = Path(p)
        vasprun = p / 'vasprun.xml'
        atoms = read(vasprun, format='vasp-xml')
        dimer_frames.append(atoms)

    write('dimer.xyz', dimer_frames)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--prefix', 
        help='time series files'
    )

    args = parser.parse_args()

    harvest_dimer_from_files('Pt2-*')
    exit()

    #symbols, data = parse_dimer_frames('./evaluated.xyz')
    #distances, energies = parse_dimer_from_files('O2-*')
    distances, energies = parse_dimer_from_files(args.prefix)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title(
        'Dimer', 
        fontsize=20, 
        fontweight='bold'
    )
    
    ax.set_xlabel('Distance [Å]', fontsize=16)
    ax.set_ylabel('Energyr [eV]', fontsize=16)

    ax.scatter(distances, energies)

    plt.savefig('dimer.png')
