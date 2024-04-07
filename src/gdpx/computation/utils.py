#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List, Callable

from ase import Atoms
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
