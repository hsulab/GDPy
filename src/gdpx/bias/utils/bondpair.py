#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple


import numpy as np

from ase import Atoms


def get_distance_and_shift(cell, positions, i, j, pbc: bool=True):
    """"""
    # FIXME: ...

    return


def compute_distance_and_shift(atoms: Atoms, neighlist, grp_a: List[int], grp_b: List[int]):
    """Find valid bond pairs between group_a and group_b.

    NOTE: We use a full-list here.

    """
    # -
    cell = atoms.get_cell(complete=True)

    # - find pairs within given distance
    bond_pairs = []
    bond_distances = []
    bond_shifts = []
    for i in grp_a:
        indices, offsets = neighlist.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if j in grp_b:
                shift = np.dot(offset, cell)
                dis = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + shift))
                bond_pairs.append(sorted([i, j]))
                bond_distances.append(dis)
                bond_shifts.append(dis)

    bond_pairs = np.array(bond_pairs, dtype=np.int32)
    bond_distances = np.array(bond_distances)
    # bond_shifts = np.array(bond_shifts)

    return bond_pairs, bond_distances, bond_shifts


if __name__ == "__main__":
    ...
