#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


def get_bond_distance_dict(unique_atomic_numbers, ratio: float = 1.0) -> dict:
    """"""
    bond_distance_dict = {}
    for i in unique_atomic_numbers:
        bond_distance_dict[(i, i)] = covalent_radii[i] * 2 * ratio
        for j in unique_atomic_numbers:
            if i == j:
                continue
            if (i, j) in bond_distance_dict:
                continue
            bond_distance_dict[(i, j)] = bond_distance_dict[(j, i)] = ratio * (
                covalent_radii[i] + covalent_radii[j]
            )

    return bond_distance_dict


def check_atomic_distances(
    atoms: Atoms,
    covalent_ratio,
    bond_distance_dict: dict,
    excluded_pairs: list = [],
    allow_isolated: bool = False,
):
    """"""
    is_valid = False

    cov_min, cov_max = covalent_ratio
    cutoff = max(bond_distance_dict.values()) * cov_max

    first_indices, second_indices, distances = neighbor_list(
        "ijd", atoms, cutoff, self_interaction=False
    )

    # Check if an atom is too far away from others
    found_isolated = False
    if not allow_isolated:
        num_atoms = len(atoms)
        for i in range(num_atoms):
            if i not in first_indices:
                found_isolated = True
                break

    # Check if two atoms are too close to each other
    if not found_isolated:
        atomic_numbers = atoms.get_atomic_numbers()
        for i, j, d in zip(first_indices, second_indices, distances):
            atomic_pair = (atomic_numbers[i], atomic_numbers[j])
            if (i, j) not in excluded_pairs:
                if d < bond_distance_dict[atomic_pair] * cov_min:
                    is_valid = False
                    break
        else:
            is_valid = True

    return is_valid


if __name__ == "__main__":
    ...
