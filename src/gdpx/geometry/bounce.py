#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs


def get_bond_distance_dict(atomic_numbers: List[int], ratio: float):
    """"""
    bond_distance_dict = dict()

    for i in atomic_numbers:
        for j in atomic_numbers:
            bond_distance_dict[(i, j)] = (covalent_radii[i] + covalent_radii[j]) * ratio

    return bond_distance_dict


def get_a_random_direction(rng):
    """"""
    rvec, rsq = np.zeros(3), 1.1
    while rsq > 1.0:
        rvec = 2 * rng.uniform(size=3) - 1.0
        rsq = np.linalg.norm(rvec)

    return rvec


def get_a_biased_direction(direction: str, rng):
    """"""
    # vec = np.array([-1.0, +1.0, +1.0]) / np.linalg.norm([-1.0, +1.0, +1.0])
    vec = get_a_random_direction(rng)

    if direction == "+x":
        axis = np.array([1, 0, 0])
    elif direction == "-x":
        axis = np.array([-1, 0, 0])
    elif direction == "+y":
        axis = np.array([0, 1, 0])
    elif direction == "-y":
        axis = np.array([0, -1, 0])
    elif direction == "+z":
        axis = np.array([0, 0, +1])
    elif direction == "-z":
        axis = np.array([0, 0, -1])
    else:
        axis = vec

    norm_along_direction = np.dot(vec, axis)
    biased_vec = (
        vec - norm_along_direction * axis + np.fabs(norm_along_direction) * axis
    )

    return biased_vec


def bounce_one_atom(
    atoms: Atoms,
    atom_index: int,
    biased_direction,
    max_disp,
    nlist,
    bond_min_dict,
    rng,
    print_func=print,
):
    """"""
    # new_atoms = copy.deepcopy(atoms)
    new_atoms = atoms

    disp_vec = get_a_biased_direction(biased_direction, rng)
    print_func(f"{disp_vec =}")

    new_atoms[atom_index].position += disp_vec * max_disp

    # repel neighbours
    nlist.update(new_atoms)

    box = new_atoms.get_cell(complete=True)

    repelled = []

    neigh_indices, neigh_offsets = nlist.get_neighbors(atom_index)
    for neigh_index, neigh_offset in zip(neigh_indices, neigh_offsets):
        vec = new_atoms.positions[atom_index] - (
            new_atoms.positions[neigh_index] + np.dot(neigh_offset, box)
        )
        dis = np.linalg.norm(vec)
        # print(neigh_index, neigh_offset, vec, dis)
        atomic_numbers = new_atoms.get_atomic_numbers()
        min_dis = bond_min_dict[
            (atomic_numbers[atom_index], atomic_numbers[neigh_index])
        ]
        if dis < min_dis:
            repelled.append(
                (
                    neigh_index,
                    new_atoms[atom_index].position
                    + min_dis * -vec / np.linalg.norm(vec),
                )
            )

    for neigh_index, neigh_position in repelled:
        # print(f"{neigh_index = }  {neigh_position = }")
        new_atoms[neigh_index].position = neigh_position

    return new_atoms


if __name__ == "__main__":
    ...
