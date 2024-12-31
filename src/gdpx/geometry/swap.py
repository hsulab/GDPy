#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional 

import numpy as np
from ase import Atoms

from .particle import translate_then_rotate
from .spatial import check_atomic_distances


def pick_one_particle(
    atoms: Atoms,
    identity_list: list,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Atoms, int, list[int]]:
    """"""
    num_entries = len(identity_list)
    picked = rng.choice(num_entries, size=1)[0]

    tag, atomic_indices = identity_list[picked]
    particle = atoms[atomic_indices]

    return particle, tag, atomic_indices  # type: ignore


def debug_swapped_positions(
    atoms: Atoms, pick_one: list[int], pick_two: list[int], prefix: str = "actual"
) -> None:
    """"""
    particle_one = atoms[pick_one]  # default copy
    assert isinstance(particle_one, Atoms)
    particle_two = atoms[pick_two]
    assert isinstance(particle_two, Atoms)

    # TODO: Deal with pbc for molecules
    cop_one = np.average(particle_one.get_positions(), axis=0)
    cop_two = np.average(particle_two.get_positions(), axis=0)
    print(
        f"{prefix}: {particle_one.get_chemical_formula():>24s} "
        + ("{:>12.4f}" * 3).format(*cop_one)
    )
    print(
        f"{prefix}: {particle_two.get_chemical_formula():>24s} "
        + ("{:>12.4f}" * 3).format(*cop_two)
    )

    return


def swap_particles_by_step(
    atoms: Atoms,
    identities: dict,
    num_swaps: int,
    bond_distance_dict: dict,
    covalent_ratio: list = [0.8, 2.0],
    intra_bond_pairs: list[tuple[int, int]]=[],
    max_attempts: Optional[int] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Optional[Atoms], str]:
    """Swap particles several times.

    Args:
        atoms: The input structure.
        identities: The dict with particles.
        num_swaps: The number of swaps.
        bond_distance_dict: The common distance for covalent bonds.
        covalent_ratio: The minimum and maxmimum ratios for covalent bond distances.
        intra_bond_pairs: The bonds ignored by distance check.
        max_attempts: The maxmimum attempts to swap.
        rng: The random number generator.
        
    Returns:
        The updated structure and information about swap successes and attempts.

    """
    # Set the maximum attempts to swap
    if max_attempts is None:
        max_attempts = num_swaps * 10

    # Get particle types and try to swap
    particle_types = list(identities.keys())
    num_particle_types = len(particle_types)

    records = []
    candidate, num_success = atoms, 0
    for _ in range(max_attempts):
        # Pick two different particles
        type_one, type_two = rng.choice(num_particle_types, size=2, replace=False)
        particle_one, tag_one, pick_one = pick_one_particle(
            candidate, identity_list=identities[particle_types[type_one]], rng=rng
        )
        particle_two, tag_two, pick_two = pick_one_particle(
            candidate, identity_list=identities[particle_types[type_two]], rng=rng
        )

        # Add pair to records and check them to avoid duplicate swaps
        tag_pair = (tag_one, tag_two) if tag_one <= tag_two else (tag_two, tag_one)
        if tag_pair in records:
            continue
        else:
            records.append(tag_pair)

        # Find particles before swap
        # debug_swapped_positions(candidate, pick_one, pick_two, prefix="origin")

        pos_one_for_restore = copy.deepcopy(particle_one.get_positions())
        pos_two_for_restore = copy.deepcopy(particle_two.get_positions())

        # TODO: Deal with pbc for molecules
        cop_one = np.average(pos_one_for_restore, axis=0)
        cop_two = np.average(pos_two_for_restore, axis=0)

        # Swap two positions with rotatation
        particle_one_ = translate_then_rotate(
            particle_one, position=cop_one, use_com=False, rng=rng
        )
        particle_two_ = translate_then_rotate(
            particle_two, position=cop_two, use_com=False, rng=rng
        )

        candidate.positions[pick_one] = particle_two_.positions
        candidate.positions[pick_two] = particle_one_.positions

        # Find particles by picked tags after swap
        # debug_swapped_positions(candidate, pick_one, pick_two, prefix="actual")

        # Check if the new structure is valid
        atomic_indices = [*pick_one, *pick_two]
        if check_atomic_distances(
            candidate,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            atomic_indices=atomic_indices,
            excluded_pairs=intra_bond_pairs,
            allow_isolated=False,
        ):
            num_success += 1
        else:
            # Restore the structure from the last step
            candidate.positions[pick_one] = pos_one_for_restore
            candidate.positions[pick_two] = pos_two_for_restore
            # debug_swapped_positions(candidate, pick_one, pick_two, prefix="before")

        if num_success == num_swaps:
            break
    else:
        ...

    if num_success > 0:
        ...
    else:
        candidate = None

    extra_info = f"[{num_success}/{max_attempts}]"

    return candidate, extra_info


if __name__ == "__main__":
    ...
