#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gdpx.utils.ClusterAndCUR import frames2features
import argparse
from pathlib import Path
from collections import Counter
from itertools import groupby
import pathlib

from typing import Union

import time

import numpy as np

from ase.io import read, write

from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...

from ase.io import read, write

from gdpx.utils.data.dpsets import find_systems_set

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

def plot_overview(
    frames, 
    ftol: float =0.05
):
    """ force tolerance 0.05 for perfectly optimised structures 
        while 0.10 for nearly optimised ones...
    """
    force_list = []
    optimised_frames = []
    for atoms in frames:
        forces = atoms.get_forces()
        maxforce = np.max(forces)
        if maxforce < ftol:
            optimised_frames.append(atoms)
        force_list.extend(forces.flatten().tolist())

    energies = [atoms.get_potential_energy() for atoms in frames]
    print(len(energies))
    opt_energies = [atoms.get_potential_energy() for atoms in optimised_frames]
    print(len(opt_energies))

    return energies, force_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        default='./test.xyz', help='xyz file'
    )
    parser.add_argument(
        '-c', '--chosen', nargs='*', required=True,
        help='chosen system, O 1 Pt 36'
    )
    parser.add_argument(
        '-nj', '--njobs', type=int,
        default=4, help='upper limit on number of directories'
    )
    parser.add_argument(
        '-ns', '--NotSave', action="store_false",
        help='save xyz for dpsets'
    )
    parser.add_argument(
        '-na', '--nall', action="store_false",
        help='not save train and test togather'
    )

    args = parser.parse_args()

    # format chosen
    if len(args.chosen) % 2 == 0:
        chosen = ''.join(args.chosen)
        print('Choose system %s' %chosen)
    else:
        raise ValueError('Wrong System Format')

    #find_systems(args.file, args.njobs)
    raw_path = Path(args.file)
    for p in raw_path.iterdir():
        if p.name == chosen:
            cur_system = p
            break
    
    print('find ', cur_system)
    if args.NotSave:
        train_xyz = pathlib.Path(cur_system.stem + '-train.xyz')
        test_xyz = pathlib.Path(cur_system.stem + '-test.xyz')
        all_xyz = pathlib.Path(cur_system.stem + '-all.xyz')
        if train_xyz.exists() and test_xyz.exists():
            print('read existed frames...')
            train_frames = read(train_xyz, ':')
            test_frames = read(test_xyz, ':')
        else:
            print('read and save frames...')
            train_frames, test_frames = find_systems_set(cur_system)
            write(train_xyz, train_frames)
            write(test_xyz, test_frames)
            all_frames = []
            all_frames.extend(train_frames)
            all_frames.extend(test_frames)
            if not args.nall:
                write(all_xyz, all_frames)
    else:
        train_frames, test_frames = find_systems_set(cur_system)

    # prepare to plot histogram
    train_energies, train_forces = plot_overview(train_frames, ftol=0.10)
    test_energies, test_forces = plot_overview(test_frames, ftol=0.10)

    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(16,12))
    axarr = axarr.flatten()
    
    num_bins = 50

    ax = axarr[0]
    n, bins, patches = ax.hist(
        train_energies, num_bins, density=False,
        label = 'Trainset'
    )
    n, bins, patches = ax.hist(
        test_energies, num_bins, density=False,
        label = 'Testset'
    )
    ax.legend()

    ax = axarr[1]
    n, bins, patches = ax.hist(
        train_forces, num_bins, density=False,
        label = 'Trainset'
    )
    n, bins, patches = ax.hist(
        test_forces, num_bins, density=False,
        label = 'Testset'
    )
    ax.legend()


    plt.savefig('ana.png')
