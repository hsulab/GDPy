#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import copy
import pathlib

from typing import Union, List, Callable

import numpy as np

from ase import Atoms
from ase.io import read, write

try:
    import dpdata
except:
    ...


def get_formula_from_atoms(atoms: Atoms) -> str:
    """Get atoms' chemical composition sorted by the alphabetic order."""

    chemical_symbols = atoms.get_chemical_symbols()
    composition = collections.Counter(chemical_symbols)
    sorted_composition = sorted(composition.items(), key=lambda x: x[0])

    return "".join([str(k) + str(v) for k, v in sorted_composition])


def convert_groups(
    names: List[str],
    groups: List[List[Atoms]],
    batchsizes: Union[List[int], int],
    type_map: List[str],
    suffix: str,
    dest_dir: Union[str, pathlib.Path] = "./",
    pfunc: Callable = print,
) -> None:
    """Dump structures to dp trainning format."""

    nsystems = len(groups)
    if isinstance(batchsizes, int):
        batchsizes = [batchsizes] * nsystems

    # --- dpdata conversion
    dest_dir = pathlib.Path(dest_dir)
    train_set_dir = dest_dir / f"{suffix}"
    if not train_set_dir.exists():
        train_set_dir.mkdir()

    sys_dirs = []

    cum_batchsizes = 0  # number of batchsizes for training
    for name, frames, batchsize in zip(names, groups, batchsizes):
        nframes = len(frames)
        nbatch = int(np.ceil(nframes / batchsize))
        pfunc(
            f"{suffix} system {name} nframes {nframes} nbatch {nbatch} batchsize {batchsize}"
        )
        # --- check composition consistent
        compositions = [get_formula_from_atoms(a) for a in frames]
        assert (
            len(set(compositions)) == 1
        ), f"Inconsistent composition {len(set(compositions))} =? 1..."
        curr_composition = compositions[0]

        cum_batchsizes += nbatch
        # --- NOTE: need convert forces to force
        frames_ = copy.deepcopy(frames)
        # check pbc
        pbc = np.all([np.all(a.get_pbc()) for a in frames_])
        for atoms in frames_:
            try:
                # NOTE: We need update info and arrays as well
                #       as some dpdata uses data from them instead of calculator
                results = atoms.calc.results
                for k, v in results.items():
                    if k in atoms.info:
                        atoms.info[k] = v
                    if k in atoms.arrays:
                        atoms.arrays[k] = v
                # - change force data
                forces = results["forces"].copy()
                del atoms.arrays["forces"]
                atoms.arrays["force"] = forces
                # - remove some keys as dpdata cannot recognise them
                #   e.g. tags, momenta, initial_charges
                keys = copy.deepcopy(list(atoms.arrays.keys()))
                for k in keys:
                    if k not in ["numbers", "positions", "force"]:
                        del atoms.arrays[k]
            except:
                ...
            finally:
                atoms.calc = None

        # --- convert data
        write(train_set_dir / f"{name}-{suffix}.xyz", frames_)
        dsys = dpdata.MultiSystems.from_file(
            train_set_dir / f"{name}-{suffix}.xyz",
            fmt="quip/gap/xyz",
            type_map=type_map,
        )
        # NOTE: this function create dir with composition and overwrite files
        #       so we need separate dirs...
        sys_dir = train_set_dir / name
        if sys_dir.exists():
            raise FileExistsError(f"{sys_dir} exists. Please check the dataset.")
        else:
            dsys.to_deepmd_npy(train_set_dir / "_temp")  # prec, set_size
            (train_set_dir / "_temp" / curr_composition).rename(sys_dir)
            if not pbc:
                with open(sys_dir / "nopbc", "w") as fopen:
                    fopen.write("nopbc\n")
        sys_dirs.append(sys_dir)
    (train_set_dir / "_temp").rmdir()

    return cum_batchsizes, sys_dirs


if __name__ == "__main__":
    ...
