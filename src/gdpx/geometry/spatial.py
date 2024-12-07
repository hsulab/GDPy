#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

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


def check_pair_distances(
    pairs,
    distances,
    chemical_numbers,
    covalent_ratio,
    bond_distance_dict,
    excluded_pairs,
):
    """"""
    cov_min, cov_max = covalent_ratio

    is_valid = False
    for p, d in zip(pairs, distances):
        i, j = p
        atomic_pair = (chemical_numbers[i], chemical_numbers[j])
        if (i, j) not in excluded_pairs:
            if d < bond_distance_dict[atomic_pair] * cov_min:
                is_valid = False
                break
    else:
        is_valid = True

    return is_valid


def check_atomic_distances(
    atoms: Atoms,
    *,
    covalent_ratio: list,
    bond_distance_dict: dict,
    excluded_pairs: list = [],
    forbidden_pairs: list = [],
    allow_isolated: bool = False,
) -> bool:
    """Check if inter-atomic distances are valid based on some criteria.

    Note:
        The `bond_distance_dict` should be like {(1,8): 0.90, (8,1): 0.90},
        the `excluded_pairs` has the atomic indices like [(1,2), (2,1)], and
        `forbidden_pairs` has the atomic-number pairs like [(8,8)].

    Args:
        atoms: The input structure.
        covalent_ratio: Two-entry list with the minimum and the maximum ratio of the covalent bond.
        bond_distance_dict: A dict with the normal covalent bond distance.
        excluded_pairs: The atomic pairs not considered in check.
        forbidden_pairs: The forbidden atomic pairs.
        allow_isolated: Whether allow atoms no neighbours to exist.

    Returns:
        Whether the structure is valid.

    """
    is_valid = False

    cov_min, cov_max = covalent_ratio
    cutoff = max(bond_distance_dict.values()) * cov_max

    first_indices, second_indices, distances = neighbor_list(
        "ijd", atoms, cutoff, self_interaction=False
    )

    # check there are at least one neighbour pair
    num_first_indices = len(set(first_indices))
    if num_first_indices != len(atoms):
        # This situation includes no pairs
        # or some atoms have no neighbours even with a larger cutoff
        if allow_isolated:
            is_valid = True
        return is_valid

    # first_indices has been sorted so we can just groupby
    chemical_numbers = atoms.get_atomic_numbers()
    for _, v in itertools.groupby(
        zip(first_indices, second_indices, distances), key=lambda p: p[0]
    ):
        found_isolated, found_too_close, found_forbidden = True, False, False
        for i, j, d in v:
            atomic_pair = (chemical_numbers[i], chemical_numbers[j])
            if (i, j) not in excluded_pairs:
                if d < bond_distance_dict[atomic_pair] * cov_min:
                    found_too_close = True
                    break
                elif d < bond_distance_dict[atomic_pair] * cov_max:
                    found_isolated = False
                    if atomic_pair in forbidden_pairs:
                        found_forbidden = True
                        break
                else:
                    ...
        else:
            # Not too close and we need check isolated
            if found_isolated and not allow_isolated:
                break
            else:
                # both good for too_close or isolated
                # move to check next atom
                ...
        if found_too_close:
            break
        if found_forbidden:
            break
    else:
        is_valid = True

    return is_valid


if __name__ == "__main__":
    ...
