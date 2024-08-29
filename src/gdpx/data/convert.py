#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from .. import config

"""Functions to convert different training datasets.
"""


def traverse_matched_dirs(wdir, pattern: str = "*.xyz"):
    """Traverse directories recursively to find those with matched files."""
    data_dirs = []

    def recursive_traverse(wdir):
        for p in wdir.iterdir():
            if p.is_dir():
                matched_paths = list(p.glob(pattern))
                if len(matched_paths) > 0:
                    data_dirs.append(p)
                recursive_traverse(p)
            else:
                ...
        return

    recursive_traverse(wdir)

    return data_dirs


def convert_dataset(pinp, custom_type_list=None):
    """"""
    pinp = pathlib.Path(pinp).resolve()
    config._print(str(pinp))

    matched_dirs = traverse_matched_dirs(pinp, "set.*")
    config._print(f"{matched_dirs =}")

    odir = pathlib.Path.cwd() / "converted" # FIXME: usd gdp -d ?
    if not odir.exists():
        odir.mkdir(parents=True)
    else:
        raise FileExistsError("")

    custom_type_list = ["O", "H"] # FIXME: ...

    num_frames = 0
    for p in matched_dirs:
        curr_frames = convert_dpset(p, custom_type_list)
        num_curr_frames = len(curr_frames)
        config._print(f"{num_curr_frames =} at {p.relative_to(pinp)}")
        ppp = odir/f"{p.parent.parent.name}_{p.parent.name}.xyz"
        config._print(ppp.name)
        write(ppp, curr_frames)

    return


def convert_dpset(pinp, custom_type_list=None):
    """Convert dataset..."""
    pinp = pathlib.Path(pinp).resolve()

    # type.raw type_map.raw nopbc
    type_digits = np.loadtxt(pinp / "type.raw", dtype=int)
    if not (pinp / "type_map.raw").exists():
        # Use custom type_list
        if custom_type_list is not None:
            type_list = custom_type_list
        else:
            raise RuntimeError(f"No type_map.raw in `{str(pinp)}`.")
    else:
        type_list = np.loadtxt(pinp / "type_map.raw", dtype=str)
    chemical_symbols = [type_list[x] for x in type_digits]
    # print(chemical_symbols)

    # box.npy  coord.npy  energy.npy  force.npy  virial.npy
    frames = []
    set_dirs = sorted(list(pinp.glob("set.*")))
    for p in set_dirs:
        box = np.load(p / "box.npy")
        coord = np.load(p / "coord.npy")
        energy = np.load(p / "energy.npy")
        force = np.load(p / "force.npy")
        if not (p / "virial.npy").exists():
            virial = None
        else:
            virial = np.load(p / "virial.npy")  # unit: eV
        nframes = box.shape[0]
        curr_frames = []
        for i in range(nframes):
            atoms = Atoms(
                chemical_symbols,
                positions=coord[i].reshape(-1, 3),
                cell=box[i].reshape(3, 3),
                pbc=True,
            )
            results = dict(
                energy=energy[i],
                forces=force[i].reshape(-1, 3),
            )
            if virial is not None:
                stress = -0.5 * (virial[i] + virial[i].T) / atoms.get_volume()
                results["stress"] = stress[[0, 4, 8, 5, 2, 1]]  # unit: eV/Ang^3
            calc = SinglePointCalculator(atoms, **results)
            atoms.calc = calc
            curr_frames.append(atoms)
        # write("./xxx.xyz", frames)
        frames.extend(curr_frames)
    # write("./xxx.xyz", frames)

    return frames


if __name__ == "__main__":
    ...
