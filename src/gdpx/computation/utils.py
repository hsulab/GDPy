#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List, Callable

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator


def copy_minimal_frames(prev_frames: List[Atoms]):
    """Copy atoms without extra information.

    Do not copy atoms.info since it is a dict and does not maitain order.

    """
    curr_frames, curr_info = [], []
    for prev_atoms in prev_frames:
        # - copy geometry
        curr_atoms = Atoms(
            symbols=copy.deepcopy(prev_atoms.get_chemical_symbols()),
            positions=copy.deepcopy(prev_atoms.get_positions()),
            cell=copy.deepcopy(prev_atoms.get_cell(complete=True)),
            pbc=copy.deepcopy(prev_atoms.get_pbc()),
        )
        curr_frames.append(curr_atoms)
        # - save info
        confid = prev_atoms.info.get("confid", -1)
        dynstep = prev_atoms.info.get("step", -1)
        prev_wdir = prev_atoms.info.get("wdir", "null")
        curr_info.append((confid, dynstep, prev_wdir))

    return curr_frames, curr_info


def make_clean_atoms(atoms_: Atoms, results: dict = None):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc()),
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms


def create_single_point_calculator(
    atoms_sorted: Atoms,
    resort: List[int],
    calc_name: str,
    properties=["energy", "forces", "free_energy", "stress"],
    print_func: Callable = print,
    debug_func: Callable = print,
):
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


def parse_type_list(atoms):
    """parse type list for read and write structure of lammps"""
    # elements
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort()  # by alphabet

    return type_list


def get_composition_from_atoms(atoms):
    """"""
    from collections import Counter

    chemical_symbols = atoms.get_chemical_symbols()
    composition = Counter(chemical_symbols)
    sorted_composition = sorted(composition.items(), key=lambda x: x[0])

    return sorted_composition


def get_formula_from_atoms(atoms):
    """"""
    from collections import Counter

    chemical_symbols = atoms.get_chemical_symbols()
    composition = Counter(chemical_symbols)
    sorted_composition = sorted(composition.items(), key=lambda x: x[0])

    return "".join([str(k) + str(v) for k, v in sorted_composition])


if __name__ == "__main__":
    ...
