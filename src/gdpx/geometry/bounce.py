#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Callable

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


def get_a_random_direction(rng: np.random.Generator) -> np.ndarray:
    """Get a random direction vector inside a unit sphere."""
    rvec, rsq = np.zeros(3), 1.1
    while rsq > 1.0:
        rvec = 2 * rng.uniform(size=3) - 1.0
        rsq = np.linalg.norm(rvec)

    return rvec


def get_a_biased_direction(direction: str, rng: np.random.Generator) -> np.ndarray:
    """Get a random direction vector biased along a specified axis.

    The final vector will be in the same hemisphere as the specified axis.

    Args:
        direction: The biased direction, can be "+x", "-x", "+y", "-y", "+z", "-z", or "" for random.
        rng: A random number generator.

    Returns:
        A biased direction vector.

    """
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
    biased_vec = vec - norm_along_direction * axis + np.fabs(norm_along_direction) * axis

    return biased_vec


def bounce_one_atom(
    atoms: Atoms,
    atom_index: int,
    biased_direction: str,
    max_disp: float,
    strength: float,
    nlist: NeighborList,
    covalent_ratio: tuple[float, float],
    bond_distance_dict: dict,
    rng: np.random.Generator,
    print_func: Callable = print,
) -> tuple[Atoms, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Bounce one atom and repel its neighbours if they are too close.

    Args:
        atoms: The ASE Atoms object.
        atom_index: The index of the atom to be bounced.
        biased_direction: The biased direction for the bounce.
        max_disp: The maximum displacement for the bounce.
        strength: The maximum repulsion strength.
        nlist: The neighbor list for the atoms.
        covalent_ratio: The covalent distance ratio (min, max).
        bond_distance_dict: The bond distance dictionary.
        rng: A random number generator.
        print_func: A function to print debug information.

    Returns:
        A tuple of the new Atoms object and a list of repelled neighbours.

    """
    # Get a random displacement vector
    new_atoms = atoms

    disp_vec = get_a_biased_direction(biased_direction, rng)
    print_func(f"{disp_vec =}")

    prev_pos = copy.deepcopy(new_atoms[atom_index].position)
    curr_pos = prev_pos + disp_vec * max_disp
    new_atoms[atom_index].position = curr_pos

    bounced = [(atom_index, prev_pos, curr_pos)]

    # Find neighbours should be repelled
    nlist.update(new_atoms)

    box = new_atoms.get_cell(complete=True)

    repelled = []

    cov_min = covalent_ratio[0]

    neigh_indices, neigh_offsets = nlist.get_neighbors(atom_index)
    for neigh_index, neigh_offset in zip(neigh_indices, neigh_offsets):
        vec = new_atoms.positions[atom_index] - (new_atoms.positions[neigh_index] + np.dot(neigh_offset, box))
        dis = np.linalg.norm(vec)
        atomic_numbers = new_atoms.get_atomic_numbers()
        min_dis = bond_distance_dict[(atomic_numbers[atom_index], atomic_numbers[neigh_index])] * cov_min
        if dis < min_dis:
            prev_pos = copy.deepcopy(new_atoms[neigh_index].position)
            disp_vec = -vec / np.linalg.norm(vec)
            curr_pos = prev_pos + disp_vec * (min_dis - dis) * strength
            repelled.append(
                (
                    neigh_index,
                    prev_pos,
                    curr_pos,
                )
            )

    # Update positions of repelled neighbours
    for neigh_index, _, neigh_position_curr in repelled:
        new_atoms[neigh_index].position = neigh_position_curr

    return new_atoms, bounced + repelled


if __name__ == "__main__":
    ...
