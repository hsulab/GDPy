#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import copy
import pathlib
from typing import Callable, Union

import numpy as np
from ase import Atoms
from ase.io import write

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
    names: list[str],
    groups: list[list[Atoms]],
    batchsizes: Union[list[int], int],
    type_map: list[str],
    suffix: str,
    dest_dir: Union[str, pathlib.Path] = "./",
    print_func: Callable = print,
) -> tuple[int, list[Union[pathlib.Path, None]]]:
    """Dump structures to deepmd trainning format.

    Args:
        names: names of the systems.
        groups: list of systems, each system is a list of Atoms objects.
        batchsizes: batchsizes for each system, or an integer for all systems.
        type_map: list of chemical symbols.
        suffix: suffix for the dataset, e.g. "train", "valid", "test".
        dest_dir: destination directory to store the dataset.
        print_func: function to print messages, default is print.

    Returns:
        cum_batchsizes: cumulative batchsizes for all systems.
        sys_dirs: list of system directories or None if a system is an empty list,
            each directory contains the deepmd dataset for a system.

    """
    # Convert batchsizes to a list
    num_systems = len(groups)
    if isinstance(batchsizes, int):
        batchsizes = [batchsizes] * num_systems

    # Write atoms objectss to deepmd sets.
    dest_dir = pathlib.Path(dest_dir)
    set_dir = dest_dir / f"{suffix}"
    if not set_dir.exists():
        set_dir.mkdir()

    sys_dirs = []

    cum_batchsizes = 0  # number of batchsizes for training
    for name, frames, batchsize in zip(names, groups, batchsizes):
        nframes = len(frames)
        nbatch = int(np.ceil(nframes / batchsize))

        # check composition consistency
        compositions = [get_formula_from_atoms(a) for a in frames]
        num_compositions = len(set(compositions))
        if num_compositions == 0:
            print_func(f"skip {suffix} system {name} nframes {nframes} nbatch {nbatch} batchsize {batchsize}")
            sys_dirs.append(None)
            continue
        else:
            if num_compositions != 1:
                raise RuntimeError(f"Inconsistent composition {num_compositions} =? 1...")
        curr_composition = compositions[0]

        print_func(f"{suffix} system {name} nframes {nframes} nbatch {nbatch} batchsize {batchsize}")

        cum_batchsizes += nbatch

        # convert forces to force
        frames_ = copy.deepcopy(frames)

        # check pbc
        pbc = np.all([np.all(a.get_pbc()) for a in frames_])
        for i, atoms in enumerate(frames_):
            try:
                # We need update info and arrays as well as some dpdata uses data from them instead of calculator.
                results = copy.deepcopy(atoms.calc.results)
                for k, v in results.items():
                    if k in atoms.info:
                        atoms.info[k] = v
                    if k in atoms.arrays:
                        atoms.arrays[k] = v
                # make sure atoms has at least energy and optional free_energy
                for k in ["energy", "free_energy"]:
                    if k in results.keys():
                        atoms.info[k] = results[k]
                    else:
                        ...
                assert "energy" in atoms.info, f"No energy in atoms.info `{i}  {atoms}`!"
                # make sure atoms has forces, and convert it to force
                forces = copy.deepcopy(results["forces"])
                atoms.arrays["force"] = forces
                assert "force" in atoms.arrays, f"No force in atoms.info `{i}  {atoms}`!"
                # make sure atoms has forces, and convert it to force
                # remove some keys as dpdata cannot recognise them
                # e.g. tags, momenta, initial_charges
                keys = copy.deepcopy(list(atoms.arrays.keys()))
                for k in keys:
                    if k not in ["numbers", "positions", "force"]:
                        del atoms.arrays[k]
            except:
                ...
            finally:
                atoms.calc = None

        # convert atoms objects to deepmd sets
        write(set_dir / f"{name}-{suffix}.xyz", frames_)
        dsys = dpdata.MultiSystems.from_file(
            set_dir / f"{name}-{suffix}.xyz",
            fmt="quip/gap/xyz",
            type_map=type_map,
        )
        (set_dir / f"{name}-{suffix}.xyz").unlink()

        # The below function create a dir based on its composition and overwrite files,
        # so we need separate dirs in case we have different systems with the same composition.
        sys_dir = set_dir / name
        if sys_dir.exists():
            raise FileExistsError(f"{sys_dir} exists. Please check the dataset.")
        else:
            dsys.to_deepmd_npy(set_dir / "_temp")  # prec, set_size
            (set_dir / "_temp" / curr_composition).rename(sys_dir)
            if not pbc:
                with open(sys_dir / "nopbc", "w") as fopen:
                    fopen.write("nopbc\n")

        sys_dirs.append(sys_dir)

    # Cleanup _temp directory
    if (set_dir / "_temp").exists():
        (set_dir / "_temp").rmdir()

    return cum_batchsizes, sys_dirs


if __name__ == "__main__":
    ...
