#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

"""Functions to convert different training datasets.
"""


def convert_dataset(pinp, format="dp"):
    """Convert dataset..."""
    pinp = pathlib.Path(pinp).resolve()
    print(pinp)

    # type.raw type_map.raw nopbc
    type_digits = np.loadtxt(pinp/"type.raw", dtype=int)
    type_list = np.loadtxt(pinp/"type_map.raw", dtype=str)
    chemical_symbols = [type_list[x] for x in type_digits]
    print(chemical_symbols)

    # box.npy  coord.npy  energy.npy  force.npy  virial.npy
    frames = []
    set_dirs = sorted(list(pinp.glob("set.*")))
    for p in set_dirs:
        box = np.load(p/"box.npy")
        coord = np.load(p/"coord.npy")
        energy = np.load(p/"energy.npy")
        force = np.load(p/"force.npy")
        if not (p/"virial.npy").exists():
            virial = None
        else:
            virial = np.load(p/"virial.npy") # unit: eV
        nframes = box.shape[0]
        curr_frames = []
        for i in range(nframes):
            atoms = Atoms(
                chemical_symbols, positions=coord[i].reshape(-1, 3), 
                cell=box[i].reshape(3, 3), pbc=True
            )
            results = dict(
                energy=energy[i], forces=force[i].reshape(-1, 3),
            )
            if virial is not None:
                stress = -0.5 * (virial[i] + virial[i].T) / atoms.get_volume()
                results["stress"] = stress[[0, 4, 8, 5, 2, 1]] # unit: eV/Ang^3
            calc = SinglePointCalculator(atoms, **results)
            atoms.calc = calc
            curr_frames.append(atoms)
        #write("./xxx.xyz", frames)
        frames.extend(curr_frames)
    write("./xxx.xyz", frames)

    return frames


if __name__ == "__main__":
    ...