#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import Counter
from itertools import groupby

import time

import numpy as np

from ase.io import read, write

from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

"""
split total dataset into systemwise one
report data distribution

two ways

from xyz

from dp set
"""

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    return

def check_atoms(atoms, chosen_counts):
    symbol_counts = Counter(atoms.get_chemical_symbols())
    # find 
    if symbol_counts == chosen_counts:
        return atoms
    else:
        return None

def find_systems(xyz_path, njobs=4):
    chosen_frames = []
    chosen_counts = {'O': 1, 'Pt': 36} # O on 3x3 Pt(111) surface

    name = ''.join([str(key)+str(value) for key, value in chosen_counts.items()])
    print(name)
    out_xyz = Path(name + '.xyz')

    #if out_xyz.exists():
    #    chosen_frames = read(out_xyz, ':')
    #else:
    #    frames = read(xyz_path, ':')
    frames = read(xyz_path, ':')
    
    st = time.time()
    chosen_frames = Parallel(n_jobs=njobs)(
        delayed(check_atoms)(atoms, chosen_counts) for atoms in frames
    )
    et = time.time()
    print('time: ', et - st)
    chosen_frames = [atoms for atoms in chosen_frames if atoms is not None]

    energies = []
    for atoms in chosen_frames:
        energies.append(atoms.get_potential_energy())

    write(name+'.xyz', chosen_frames)

def find_systems_set(raw_path, chosen, njobs=4):
    raw_path = Path(raw_path)
    for p in raw_path.iterdir():
        # slower than re, re.findall(r'[0-9]+|[a-z]+',s)
        # cur_system = [''.join(list(g)) for k, g in groupby(p.name, key=lambda x: x.isdigit())] 
        if p.name == chosen:
            cur_system = p
            break
    
    set_dirs = []
    for p in cur_system.glob('set*'):
        set_dirs.append(p)
    set_dirs.sort()

    #inverse_type_map = dict(zip())
    inverse_type_map = {0: 'O', 1: 'Pt'}
    atype = np.loadtxt(cur_system / 'type.raw', dtype=int)
    print(atype)
    chemical_symbols = [inverse_type_map[a] for a in atype]

    # train data
    total_frames = []
    boxes, coords, energies, forces = [], [], [], []
    for set_dir in set_dirs[:-1]:
        #boxes.extend( np.load(set_dir / 'box.npy') )
        #coords.extend( np.load(set_dir / 'coord.npy') )
        #energies.extend( np.load(set_dir / 'energy.npy') )
        #forces.extend( np.load(set_dir / 'force.npy') )
        boxes = np.load(set_dir / 'box.npy')
        coords = np.load(set_dir / 'coord.npy')
        energies = np.load(set_dir / 'energy.npy')
        forces = np.load(set_dir / 'force.npy')
        nframes = boxes.shape[0]
        print('nframes', nframes)
        for i in range(nframes):
            cell = boxes[i,:].reshape(3,3)
            positions = coords[i,:].reshape(-1,3)
            atoms = Atoms(symbols=chemical_symbols, positions=positions, cell=cell)
            results = {'energy': energies[i], 'forces': forces[i,:].reshape(-1,3)}
            spc = SinglePointCalculator(atoms, **results)
            atoms.calc = spc
            total_frames.append(atoms)
    write(cur_system.name+'-train.xyz', total_frames)
    
    # form atoms

    return

def plot_stat():
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    scatter_hist(energies, energies, ax, ax_histx, ax_histy)

    plt.savefig('miaow.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir',
        default='./', help='vasp calculation directory'
    )
    parser.add_argument(
        '-f', '--file',
        default='./test.xyz', help='xyz file'
    )
    parser.add_argument(
        '-c', '--chosen', nargs='*', required=True,
        help='chosen system, O 1 Pt 36'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='vasp_0_*', help='vasp directory name pattern'
    )
    parser.add_argument(
        '-nj', '--njobs', type=int,
        default=4, help='upper limit on number of directories'
    )

    args = parser.parse_args()

    # format chosen
    if len(args.chosen) % 2 == 0:
        args.chosen = ''.join(args.chosen)
        print('Choose system %s' %args.chosen)
    else:
        raise ValueError('Wrong System Format')

    #find_systems(args.file, args.njobs)
    find_systems_set(args.file, args.chosen, args.njobs)