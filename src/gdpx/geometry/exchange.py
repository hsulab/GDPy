#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
import itertools
from typing import Callable, Optional

import numpy as np
from ase import Atoms

from gdpx.utils.atoms_tags import reassign_tags_by_species

from .particle import translate_then_rotate
from .spatial import check_atomic_distances


def remove_one_particle(
    atoms: Atoms,
    identities: dict,
    species: str,
    sort_tags: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Atoms, str]:
    """Remove one particle from the given atoms.

    Args:
        atoms: The input structure.
        identities: A dict with particles names and atomic indices.
        species: The particle name.
        sort_tags: Whether sort atoms by tags.
        rng: The random number generator.

    Returns:
        The new structure and the auxiliary information.

    """
    num_particles = len(identities[species])
    selected = rng.choice(range(num_particles), replace=False)
    selected_indices = identities[species][selected][1]
    del atoms[selected_indices]

    if sort_tags:
        atoms = reassign_tags_by_species(atoms)

    return atoms, f"remove_{species}_{selected_indices}"


def insert_one_particle(
    atoms: Atoms,
    particle: Atoms,
    region,
    covalent_ratio,
    bond_distance_dict,
    particle_tag: Optional[int] = None,
    sort_tags: bool = True,
    max_attempts: int = 100,
    check_distance_func: Optional[Callable] = check_atomic_distances,
    copy_atoms: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Optional[Atoms], str]:
    """"""
    # Set the tag for the inserted particle,
    # which should not be used in atoms.
    if particle_tag is None:
        particle_tag = int(np.max(atoms.get_tags()) + 17)
    particle.set_tags(particle_tag)

    # Check if we should check neighbour distances
    num_atoms = len(atoms)

    if check_distance_func is not None:
        # Avoid distance check in the substrate and the particle to insert
        intra_bond_pairs = list(itertools.permutations(range(0, num_atoms), 2))
        intra_bond_pairs.extend(list(itertools.permutations(range(num_atoms, num_atoms + len(particle)), 2)))
        # We only check bond distances form by atoms in the particle,
        # since the existing atoms may not statisfy our distance criteria.
        atomic_indices = list(range(num_atoms, num_atoms + len(particle)))
        # Build the function
        post_func = functools.partial(
            check_distance_func,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            atomic_indices=atomic_indices,
            excluded_pairs=intra_bond_pairs,
            allow_isolated=False,
        )
    else:
        post_func = lambda _: True

    # Try inserting
    num_attempts = 0
    if copy_atoms:
        candidate = atoms + particle
    else:  # A revert is necessary if insert is used in MC.
        candidate = atoms
        candidate.extend(particle)
    region.preprocess(candidate)
    for iattempt in range(max_attempts):
        if copy_atoms and (len(atoms) != num_atoms):
            # Make sure we have not messed up with the substrate
            raise Exception(f"Expecting {num_atoms} but got {len(atoms)}.")
        position = region.get_random_positions(size=1, rng=rng)[0]
        new_particle = copy.deepcopy(particle)
        new_particle = translate_then_rotate(new_particle, position=position, use_com=True, rng=rng)
        candidate.positions[num_atoms:] = new_particle.positions
        if post_func(candidate):
            num_attempts = iattempt + 1
            break
    else:
        candidate = None
        num_attempts = max_attempts

    chemical_formula = particle.get_chemical_formula()
    state = "success"
    if candidate is not None:
        if sort_tags:
            candidate = reassign_tags_by_species(candidate)
    else:
        state = "failure"

    return candidate, f"insert_{chemical_formula}_{state}_{num_attempts}"


if __name__ == "__main__":
    ...
