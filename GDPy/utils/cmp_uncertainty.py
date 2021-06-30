#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt

from ase.io import read, write

from GDPy.calculator.dp import DP
from GDPy.utils.comparasion import parity_plot

# frames = read('./PtO/PtO_ALL.xyz', ':')
frames = read('./PtO/calculated_0.xyz', ':')
print(len(frames))

dft_energies, dft_forces = [], []
for atoms in frames:
    dft_energies.append(atoms.get_potential_energy())
    dft_forces.append(atoms.get_forces())
dft_energies = np.array(dft_energies)
dft_forces = np.array(dft_forces)

def calculate():
    calc = DP(
        model = [
            '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-0/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-1/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-2/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-3/graph.pb'
        ],
        type_dict = {'O': 0, 'Pt': 1}
    )
    
    all_energies, all_forces = [], []
    all_fstdvar = []
    for atoms in frames:
        calc.reset()
        atoms.calc = calc
        en = atoms.get_potential_energy()
        all_energies.append(en)
        forces = atoms.get_forces()
        all_forces.append(forces)
        all_fstdvar.append(atoms.calc.results['forces_stdvar'])
    
    all_energies = np.array(all_energies)
    np.save('energy.npy', all_energies)
    
    all_forces = np.array(all_forces)
    np.save('forces.npy', all_forces)

    all_fstdvar = np.array(all_fstdvar)
    np.save('fstdvar.npy', all_fstdvar)

#calculate()

dp_energies = np.load('energy.npy')
dp_forces = np.load('forces.npy')
dp_fstdvar = np.load('fstdvar.npy')


fig, axarr = plt.subplots(
    nrows=2, ncols=2, 
    gridspec_kw={'hspace': 0.3}, figsize=(16,16)
)
axarr = axarr.flat[:]

natoms = 32
parity_plot(
    ('DP',dp_energies/natoms), ('DFT',dft_energies/natoms), axarr[0],
    prop = ('energy','[eV/atom]')
)
parity_plot(
    ('DP',dp_forces), ('DFT',dft_forces), axarr[1],
    prop = ('force','[eV/AA]')
)

parity_plot(
    ('Uncertainty',dp_fstdvar), ('Error',np.fabs(dp_forces-dft_forces)), axarr[3],
    prop = ('force','[eV/AA]')
)

plt.savefig('pp.png')


if __name__ == '__main__':
    pass
