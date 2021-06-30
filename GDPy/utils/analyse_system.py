#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt
plt.style.use('presentation')

from GDPy.utils.comparasion import force_plot
from ase.io import read, write

def check_convergence(forces):
    """
    forces nframes*(3*natoms) array
    """
    max_forces = np.max(forces, axis=1)

    return max_forces

frames = read('/users/40247882/scratch2/analyse-data/O1Pt36-train.xyz', ':')

optimised_frames = []
for atoms in frames:
    forces = atoms.get_forces()
    maxforce = np.max(forces)
    if maxforce < 0.05:
        optimised_frames.append(atoms)

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(16,12))
axarr = axarr.flatten()

ax = axarr[0]

energies = [atoms.get_potential_energy() for atoms in frames]
print(len(energies))
opt_energies = [atoms.get_potential_energy() for atoms in optimised_frames]
print(len(opt_energies))

num_bins = 50
n, bins, patches = ax.hist(energies, num_bins, density=False)
#n, bins, patches = ax.hist(opt_energies, num_bins, density=False)

plt.savefig('ana.png')


if __name__ == '__main__':
    pass