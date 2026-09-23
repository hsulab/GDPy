#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase import Atoms
from ase.geometry import find_mic
from ase.mep import NEBTools


def compute_rxn_coords(frames: list[Atoms]):
    """Compute reaction coordinates."""
    # Copy positions to avoid change atoms positions and lost energy properties.
    nframes = len(frames)
    natoms = len(frames[0])
    coordinates = np.zeros((nframes, natoms, 3))
    for i, _ in enumerate(frames):
        coordinates[i, :, :] = copy.deepcopy(frames[i].get_positions())

    # TODO: If the cell is changed?
    rxn_coords = []
    cell = frames[0].get_cell(complete=True)
    for i in range(1, nframes):
        prev_positions = coordinates[i - 1]
        curr_positions = coordinates[i]
        shift = curr_positions - prev_positions
        curr_vectors, _ = find_mic(shift, cell, pbc=True)
        coordinates[i] = prev_positions + curr_vectors
        rxn_coords.append(np.linalg.norm(curr_vectors))

    rxn_coords = np.cumsum(rxn_coords)
    rxn_coords = np.hstack(([0.0], rxn_coords))

    return rxn_coords


def plot_mep(wdir: pathlib.Path, images: list[Atoms]):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    nbt = NEBTools(images=images)
    nbt.plot_band(ax=ax)

    fig.savefig(wdir / "neb.png")

    plt.close()

    return


def plot_bands(wdir: pathlib.Path, images: list[Atoms], num_images_per_band: int):
    """"""
    num_frames = len(images)

    nbands = int(num_frames / num_images_per_band)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    for i in range(nbands):
        nbt = NEBTools(images=images[i * num_images_per_band : (i + 1) * num_images_per_band])
        nbt.plot_band(ax=ax)

    fig.savefig(wdir / "bands.png")

    plt.close()

    return


if __name__ == "__main__":
    ...
