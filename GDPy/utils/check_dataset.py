#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt

print(mpl.get_configdir())
plt.style.use('presentation')

import deepmd.DeepPot as DeepPot

from GDPy.utils.comparasion import parity_plot, parity_plot_dict, PropInfo

"""
properties we may be interested in

maximum force if it is a local minimum

energy and force distribution (both train and test)

use DeepEval to check uncertainty

"""

type_map = {"O": 0, "Pt": 1}
inverse_type_map = {0: "O", 1: "Pt"}

def check_convergence(forces):
    """
    forces nframes*(3*natoms) array
    """
    max_forces = np.max(forces, axis=1)

    return max_forces

def transform_forces(symbols, forces):
    """"""
    #print(symbols)
    forces = np.array(forces)
    print(forces.shape)
    elemental_forces = {}
    for elem in type_map.keys():
        elemental_forces[elem] = []
    for idx, symbol in enumerate(symbols):
        print(inverse_type_map[symbol])
        elemental_forces[inverse_type_map[symbol]].extend(
            list(forces[:,3*idx:3*(idx+1)].flatten())
        )
    for elem in elemental_forces.keys():
        elemental_forces[elem] = np.array(elemental_forces[elem])
        print(elemental_forces[elem].shape)
    elemental_forces = elemental_forces

    return elemental_forces

def plot_hist(ax, data, xlabel, ylabel):
    num_bins = 50
    n, bins, patches = ax.hist(data, num_bins, density=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return

raw_dirs = Path('../oxides/gdp-main/it-0011/raw_data')

# find system
sys_map = {"O": 1, "Pt": 36}
sys_name = []
for k, v in sys_map.items():
    sys_name.extend([str(k), str(v)])
sys_name = ''.join(sys_name)

test = raw_dirs / sys_name
print(test)

natoms = np.sum(list(sys_map.values())) 

set_dirs = list(test.glob('set*'))
set_dirs.sort()

# train data
train_boxes, train_coords, train_energies, train_forces = [], [], [], []
for set_dir in set_dirs[:-1]:
    train_boxes.extend( np.load(set_dir / 'box.npy').tolist() )
    train_coords.extend( np.load(set_dir / 'coord.npy').tolist() )
    train_energies.extend( np.load(set_dir / 'energy.npy').tolist() )
    train_forces.extend( np.load(set_dir / 'force.npy').tolist() )
    #print(boxes.shape)
    #print(boxes)

# test_data
test_boxes, test_coords, test_energies, test_forces = [], [], [], []
for set_dir in set_dirs[-1:]:
    test_boxes.extend( np.load(set_dir / 'box.npy').tolist() )
    test_coords.extend( np.load(set_dir / 'coord.npy').tolist() )
    test_energies.extend( np.load(set_dir / 'energy.npy').tolist() )
    test_forces.extend( np.load(set_dir / 'force.npy').tolist() )

# read types
atype = np.loadtxt(test / 'type.raw', dtype=int)
print(atype)

# start with a square Figure
fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(16,16))
axarr = axarr.flatten()

plt.suptitle('Dataset Overview')
#plt.tight_layout()
plt.subplots_adjust(
    left=0.10, right=0.95,
    bottom=0.10, top=0.85,
    wspace=0.20, hspace=0.30
)

plot_hist(axarr[0], np.array(train_energies).flatten()/natoms, 'Energy [eV]', 'Number of Frames')
plot_hist(axarr[1], np.array(train_forces).flatten(), 'Force [eV/AA]', 'Number of Frames')
plot_hist(axarr[0], np.array(test_energies).flatten()/natoms, 'Energy [eV]', 'Number of Frames')
plot_hist(axarr[1], np.array(test_forces).flatten(), 'Force [eV/AA]', 'Number of Frames')

# use dp to check test
dp = DeepPot('../oxides/gdp-main/it-0011/ensemble/model-0/graph.pb')
dp_input = {'coords': test_coords, 'cells': test_boxes, 'atom_types': atype, 'atomic': True}
e, f, v, ae, av = dp.eval(**dp_input)

prop_info = PropInfo(xlabel='miaow', ylabel='miaow', title='xxx')
parity_plot(['DP', e.flatten()/natoms], ['DFT', np.array(test_energies).flatten()/natoms], axarr[2], ('energy', '[eV/atom]'), sys_name)
#parity_plot(['DP', f.flatten()], ['DFT', np.array(test_forces).flatten()], axarr[3], ('force', '[eV/AA]'), sys_name)
parity_plot_dict(transform_forces(atype, f.reshape(-1,natoms*3)), transform_forces(atype, test_forces), axarr[3], prop_info)

dp_input = {'coords': train_coords, 'cells': train_boxes, 'atom_types': atype, 'atomic': True}
e, f, v, ae, av = dp.eval(**dp_input)

#parity_plot(['DP', e.flatten()/natoms], ['DFT', np.array(test_energies).flatten()/natoms], axarr[4], ('energy', '[eV/atom]'), sys_name)
#parity_plot_dict(transform_forces(atype, f.reshape(-1,natoms*3)), transform_forces(atype, test_forces), axarr[5], prop_info)

plt.savefig('wang.png')


if __name__ == '__main__':
    pass
