import itertools
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, neighbor_list

from .restraints import ParsedRestraint, minimum_distance_for_pair, validate_restraint_tags


def get_bond_distance_dict(unique_atomic_numbers, ratio: float = 1.0) -> dict[tuple[int, int], float]:
    """"""
    bond_distance_dict = {}
    for i in unique_atomic_numbers:
        bond_distance_dict[(i, i)] = covalent_radii[i] * 2 * ratio
        for j in unique_atomic_numbers:
            if i == j:
                continue
            if (i, j) in bond_distance_dict:
                continue
            bond_distance_dict[(i, j)] = bond_distance_dict[(j, i)] = ratio * (covalent_radii[i] + covalent_radii[j])

    return bond_distance_dict


def check_pair_distances(
    pairs,
    distances,
    chemical_numbers,
    covalent_ratio: tuple[float, float],
    bond_distance_dict,
    excluded_pairs,
    restraints: Optional[list[ParsedRestraint]] = None,
    tags: Optional[np.ndarray] = None,
):
    """"""
    cov_min, _ = covalent_ratio

    is_valid = False
    restraints = restraints or []
    chemical_numbers = np.asarray(chemical_numbers)
    if restraints and any(restraint.scope == "inter_particle" for restraint in restraints) and tags is None:
        raise ValueError("`tags` are required when checking `inter_particle` restraints.")
    for p, d in zip(pairs, distances):
        i, j = p
        atomic_pair = (chemical_numbers[i], chemical_numbers[j])
        if (i, j) not in excluded_pairs:
            dmin = minimum_distance_for_pair(
                int(i),
                int(j),
                chemical_numbers,
                tags,
                restraints,
                bond_distance_dict[atomic_pair] * cov_min,
            )
            if d < dmin:
                is_valid = False
                break
    else:
        is_valid = True

    return is_valid


def check_atomic_distances(
    atoms: Atoms,
    *,
    covalent_ratio: tuple[float, float],
    bond_distance_dict: dict,
    atomic_indices: Optional[list[int]] = None,
    excluded_pairs: list = [],
    restraints: Optional[list[ParsedRestraint]] = None,
    allow_isolated: bool = False,
) -> bool:
    """Check if inter-atomic distances are valid based on some criteria.

    Note:
        The `bond_distance_dict` should be like {(1,8): 0.90, (8,1): 0.90},
        the `excluded_pairs` has the atomic indices like [(1,2), (2,1)].

    Args:
        atoms: The input structure.
        covalent_ratio: Two-entry list with the minimum and the maximum ratio of the covalent bond.
        bond_distance_dict: A dict with the normal covalent bond distance.
        atomic_indices: The indices of atomic centres to check.
        excluded_pairs: The atomic pairs (should be symmetric) not considered in too_close and forbidden.
        restraints: Parsed geometric restraints with optional pair-specific minima.
        allow_isolated: Whether allow atoms without neighbours to exist.

    Returns:
        Whether the structure is valid.

    """
    is_valid = False

    cov_min, cov_max = covalent_ratio
    restraints = restraints or []
    if restraints:
        validate_restraint_tags(atoms, restraints)

    chemical_numbers = atoms.get_atomic_numbers()
    tags = atoms.get_tags() if atoms.has("tags") else None
    cutoff = np.array([covalent_radii[c] for c in chemical_numbers]) * cov_max

    first_indices, second_indices, distances = neighbor_list("ijd", atoms, cutoff, self_interaction=False)

    num_atoms = len(atoms)
    if atomic_indices is None:
        atomic_indices = list(range(num_atoms))

    # check there are at least one neighbour pair
    found_isolated = False
    if not allow_isolated:
        centres_with_neighbours = set(first_indices)
        for c_i in range(num_atoms):
            if c_i not in centres_with_neighbours and c_i in atomic_indices:
                found_isolated = True
                break
        else:
            ...
    if found_isolated:
        return is_valid

    # first_indices has been sorted so we can just groupby
    for c_i, v in itertools.groupby(zip(first_indices, second_indices, distances), key=lambda p: p[0]):
        if c_i not in atomic_indices:
            continue
        found_isolated, found_too_close = True, False
        for i, j, d in v:
            atomic_pair = (chemical_numbers[i], chemical_numbers[j])
            dmin = minimum_distance_for_pair(
                int(i),
                int(j),
                chemical_numbers,
                tags,
                restraints,
                bond_distance_dict[atomic_pair] * cov_min,
            )
            dmax = bond_distance_dict[atomic_pair] * cov_max
            # We only check too_close for pairs not excluded.
            # The excluded pairs are used to determine isolation.
            if d <= dmax:
                found_isolated = False
                if (i, j) not in excluded_pairs:
                    if d < dmin:
                        found_too_close = True
                        break
                    else:  # dmin < d <= dmax
                        ...
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
    else:
        is_valid = True

    return is_valid


def check_atomic_distances_by_neighbour_list(
    atoms: Atoms,
    *,
    neighlist: NeighborList,
    atomic_indices: list[int],
    covalent_ratio: tuple[float, float],
    bond_distance_dict: dict,
    excluded_pairs: list = [],
    allow_isolated: bool = False,
):
    """Check atomic distances based on a pre-computed neighbour list.

    Args:
        neighlist: This must be bothways and no self-interactions.
        excluded_pairs: The atomic pairs (should be symmetric) not considered in too_close and forbidden.

    """
    # Some basic stuff
    cell = atoms.get_cell(complete=True)

    # Get covalent bond distance ratio
    cov_min, cov_max = covalent_ratio

    # Get chemical numbers here since some operators may change the symbol
    chemical_numbers = atoms.get_atomic_numbers()

    # Check neighbour list
    neighlist.update(atoms)

    # Check atomic distances
    is_valid = False
    for _, idx_pick in enumerate(atomic_indices):
        indices, offsets = neighlist.get_neighbors(idx_pick)
        found_too_close, found_isolated = False, True
        for ni, offset in zip(indices, offsets):
            if ni not in atomic_indices:  # Skip intra-molecular check
                distance = np.linalg.norm(atoms.positions[idx_pick] - (atoms.positions[ni] + np.dot(offset, cell)))
                atomic_pair = (chemical_numbers[idx_pick], chemical_numbers[ni])
                dmin = bond_distance_dict[atomic_pair] * cov_min
                dmax = bond_distance_dict[atomic_pair] * cov_max
                if distance <= dmin:
                    # Avoid too close bonds in molecules
                    if (idx_pick, ni) not in excluded_pairs:
                        found_too_close = True
                        break
                elif distance <= dmax:
                    found_isolated = False
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
    else:
        is_valid = True

    return is_valid
