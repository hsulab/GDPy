#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from typing import List, Callable

import numpy as np

from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase.formula import Formula
from ase.geometry import find_mic
from ase.neb import NEB, NEBTools


def convert_index_to_formula(atoms, group_indices: List[List[int]]):
    """"""
    formulae = []
    for g in group_indices:
        symbols = [atoms[i].symbol for i in g]
        formulae.append(
            Formula.from_list(symbols).format("hill")
        )
    #formulae = sorted(formulae)

    return formulae

def compute_rxn_coords(frames):
    """Compute reaction coordinates."""
    # - avoid change atoms positions and lost energy properties...
    nframes = len(frames)
    natoms = len(frames[0])
    coordinates = np.zeros((nframes, natoms, 3))
    for i, a in enumerate(frames):
        coordinates[i, :, :] = copy.deepcopy(frames[i].get_positions())

    rxn_coords = []
    cell = frames[0].get_cell(complete=True)
    for i in range(1, nframes):
        prev_positions = coordinates[i-1]
        curr_positions = coordinates[i]
        shift = curr_positions - prev_positions
        curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)
        coordinates[i] = prev_positions + curr_vectors
        rxn_coords.append(np.linalg.norm(curr_vectors))

    rxn_coords = np.cumsum(rxn_coords)
    rxn_coords = np.hstack(([0.], rxn_coords))

    return rxn_coords

def plot_mep(wdir, images):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    nbt = NEBTools(images=images)
    nbt.plot_band(ax=ax)

    plt.savefig(wdir/"neb.png")

    plt.close()

    return

def plot_bands(wdir, images, nimages: int):
    """"""
    #print([a.get_potential_energy() for a in images])
    
    nframes = len(images)

    nbands = int(nframes/nimages)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    for i in range(nbands):
        #print(f"plot_bands {i}")
        nbt = NEBTools(images=images[i*nimages:(i+1)*nimages])
        nbt.plot_band(ax=ax)

    plt.savefig(wdir/"bands.png")

    return


if __name__ == "__main__":
    ...
