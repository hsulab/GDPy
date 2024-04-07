#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pathlib

from typing import Callable, List, Tuple

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


"""This module stores several utilities that copy structures.
"""


def read_sort(
    directory: pathlib.Path, sort_fname: str = "ase-sort.dat"
) -> Tuple[List[int], List[int]]:
    """Create the sorting and resorting list from ase-sort.dat.

    If the ase-sort.dat file does not exist, the sorting is redone.

    """
    sortfile = directory / sort_fname
    if os.path.isfile(sortfile):
        sort = []
        resort = []
        with open(sortfile, "r") as fd:
            for line in fd:
                s, rs = line.split()
                sort.append(int(s))
                resort.append(int(rs))
    else:
        # warnings.warn(UserWarning, 'no ase-sort.dat')
        raise ValueError("no ase-sort.dat")

    return sort, resort


def resort_atoms_with_spc(
    atoms_sorted: Atoms,
    resort: List[int],
    calc_name: str,
    properties=["energy", "forces", "free_energy", "stress"],
    print_func: Callable = print,
    debug_func: Callable = print,
) -> Atoms:
    """Create a spc to store calc results since some atoms may share a calculator.

    TODO: Add charges and magmoms?

    """
    _print = print_func
    _debug = debug_func

    #
    atoms = atoms_sorted.copy()[resort]

    # The original calculator must have an energy property
    results = dict(
        energy=atoms_sorted.get_potential_energy(apply_constraint=False),
    )

    try:
        free_energy = atoms_sorted.get_potential_energy(
            force_consistent=True, apply_constraint=False
        )
    except:
        free_energy = None  # results["energy"]
        _print("No free_energy property.")
    if free_energy is not None:
        results["free_energy"] = free_energy

    try:
        forces = atoms_sorted.get_forces(apply_constraint=False)[resort]
    except:
        forces = None
        _print("No forces property.")
    if forces is not None:
        results["forces"] = forces

    try:
        stress = atoms_sorted.get_stress()
    except:
        stress = None
        _print("No stress property.")
    if stress is not None:
        results["stress"] = stress

    calc = SinglePointCalculator(atoms, **results)
    calc.name = calc_name
    atoms.calc = calc

    return atoms


if __name__ == "__main__":
    ...
