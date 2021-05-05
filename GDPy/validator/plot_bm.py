#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot various BM curves 
"""

from pathlib import Path
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.eos import EquationOfState 

def read_arrays(datafile):
    with open(datafile, 'r') as reader:
        lines = reader.readlines()

    lines = [line.strip().split() for line in lines if not line.startswith('#')]

    data = np.array(lines, dtype=float)

    return data

def read_eos(eos_dir):
    vols, energies = [], []
    eos_dir = Path(eos_dir)
    for p in eos_dir.glob('*'):
        vasprun = p / 'vasprun.xml'
        if vasprun.exists():
            atoms = read(vasprun, format='vasp-xml')
            vols.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
    
    return vols, energies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dft', '--dftdata', \
            help='time series files')
    parser.add_argument('-dp', '--dpdata', \
            help='time series files')
    parser.add_argument('-n', '--name', \
            help='time series files')

    args = parser.parse_args()

    composition = 'Pt'
    vols, energies = read_eos('Pt_eos')

    # plot figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title(
        composition + ' Birch-Murnaghan (Constant-Volume Optimisation)', \
        fontsize=24, fontweight='bold'
    )
    
    ax.set_xlabel('Volume/atom [Å^3/atom]', fontsize=20)
    ax.set_ylabel('Energy/atom [eV/atom]', fontsize=20)

    #ax.plot(dft_curve[:,0], dft_curve[:,1])

    ax.scatter(vols, energies, marker='*', label='DFT')
    #ax.scatter(dft[:,0], dft[:,1], marker='*', label='DFT')
    #ax.scatter(dp[:,0], dp[:,1], marker='x', label='DP')

    ax.legend(fontsize=20)

    plt.savefig('bm.png')

