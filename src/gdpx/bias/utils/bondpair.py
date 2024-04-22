#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import List, Tuple


import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii


def get_equidis_dict(bonds, ratio: float = 1.0):
    """Get bond equilibrium distance from a string input.

    Args:
        bonds: A list [C, H, O] or [(C-O), (C-H)].
        ratio: Mutiplier to the atomic covalent radii.

    Returns:
        Atomic symbols, bond pairs, equilibrium distance dict.

    """
    #: Equilibrium bond distance dict.
    if isinstance(bonds[0], str):
        symbols = bonds
        bonds = list(itertools.product(bonds, bonds))
    else:
        symbols_, bonds_ = [], []
        for i, j in bonds:
            symbols_.extend([i, j])
            bonds_.append((i, j))
            bonds_.append((j, i))
        symbols = list(set(symbols_))
        bonds = list(set(bonds_))

    radii = {s: ratio * covalent_radii[atomic_numbers[s]] for s in symbols}
    equidis_dict = {k: radii[k[0]] + radii[k[1]] for k in bonds}

    return symbols, bonds, equidis_dict


def get_distance_and_shift(cell, positions, i, j, pbc: bool = True):
    """"""
    # FIXME: ...

    return


def compute_distance_and_shift(
    atoms: Atoms, grp_a: List[int], grp_b: List[int], neighlist=None
):
    """Find valid bond pairs between group_a and group_b.

    NOTE: We use a full-list here.

    """
    # -
    cell = atoms.get_cell(complete=True)

    # - find pairs within given distance
    if neighlist is not None:
        bond_pairs = []
        bond_distances = []
        bond_shifts = []
        for i in grp_a:
            indices, offsets = neighlist.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if j in grp_b:
                    shift = np.dot(offset, cell)
                    dis = np.linalg.norm(
                        atoms.positions[i] - (atoms.positions[j] + shift)
                    )
                    bond_pairs.append(sorted([i, j]))
                    bond_distances.append(dis)
                    bond_shifts.append(dis)
        bond_pairs = np.array(bond_pairs, dtype=np.int32)
        bond_distances = np.array(bond_distances)
        # bond_shifts = np.array(bond_shifts)
    else:
        bond_pairs = np.array(list(itertools.product(grp_a, grp_b)), dtype=np.int32)
        bond_distances = np.linalg.norm(
            atoms.positions[bond_pairs[:, 0]] - atoms.positions[bond_pairs[:, 1]],
            axis=1,
        )
        bond_shifts = np.zeros((len(bond_pairs), 3))

    return bond_pairs, bond_distances, bond_shifts


def get_bond_information(
    atoms: Atoms,
    neighlist,
    eqdis_dict: dict,
    covalent_min: float,
    target_indices: List[int],
    allowed_bonds: List[Tuple[str, str]],
):
    """Find valid bond pairs and compute their strains.

    NOTE: We use a half-list here. This is used in bond-boost.

    """
    # -
    cell = atoms.get_cell(complete=True)

    # - find pairs within given distance
    bond_pairs = []
    bond_curr_distances = []
    bond_curr_shifts = []
    bond_equi_distances = []
    for i in target_indices:
        sym_i = atoms[i].symbol
        indices, offsets = neighlist.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            sym_j = atoms[j].symbol
            pair = (sym_i, sym_j)
            if pair in allowed_bonds:
                shift = np.dot(offset, cell)
                dis = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + shift))
                if dis >= eqdis_dict[pair] * covalent_min:
                    bond_pairs.append(tuple(sorted([i, j])))
                    bond_curr_distances.append(dis)
                    bond_curr_shifts.append(shift)
                    bond_equi_distances.append(eqdis_dict[pair])

    bond_pairs = bond_pairs
    bond_curr_distances = np.array(bond_curr_distances)
    bond_equi_distances = np.array(bond_equi_distances)

    return bond_pairs, bond_curr_distances, bond_curr_shifts, bond_equi_distances


if __name__ == "__main__":
    ...
