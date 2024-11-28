#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import Optional, Tuple

import numpy as np
from ase import Atoms

from gdpx.utils.atoms_tags import get_tags_per_species

from .particle import translate_then_rotate
from .spatial import check_atomic_distances


def sort_tags_by_species(atoms: Atoms) -> Atoms:
    """"""
    tags_dict = get_tags_per_species(atoms)

    # Find substrate which has tag 0
    substrate: str = ""
    num_atoms_in_substrate: int = -1
    for k, v in tags_dict.items():
        num_instances = len(v)
        if num_instances == 1 and v[0][0] == 0:
            substrate = k
            num_atoms_in_substrate = len(v[0][1])
            break
    else:
        raise RuntimeError(f"Cannot find substrate with tag 0 in `{atoms}`.")

    new_tags = [0] * num_atoms_in_substrate
    new_indices = list(range(num_atoms_in_substrate))

    current_tag = 1
    valid_keys = sorted([k for k in tags_dict.keys() if k != substrate])
    for species in valid_keys:
        for k, v in tags_dict[species]:
            new_indices.extend(v)
            new_tags.extend([current_tag] * len(v))
            current_tag += 1

    new_atoms: Atoms = atoms[new_indices]  # type: ignore
    new_atoms.set_tags(new_tags)

    # Inherit info
    new_atoms.info = copy.deepcopy(atoms.info)

    return new_atoms


def remove_one_particle(
    atoms: Atoms,
    identities: dict,
    species: str,
    sort_tags: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[Atoms, str]:
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
        atoms = sort_tags_by_species(atoms)

    return atoms, f"remove_{species}_{selected_indices}"


def insert_one_particle(
    atoms: Atoms,
    particle: Atoms,
    region,
    covalent_ratio,
    bond_distance_dict,
    sort_tags: bool = True,
    max_attempts: int = 100,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[Optional[Atoms], str]:
    """"""
    particle_tag = int(np.max(atoms.get_tags()) + 17)
    particle.set_tags(particle_tag)

    # Avoid distance check in the substrate and the particle to insert
    num_atoms = len(atoms)
    intra_bond_pairs = list(itertools.permutations(range(0, num_atoms), 2))
    intra_bond_pairs.extend(
        list(itertools.permutations(range(num_atoms, num_atoms + len(particle)), 2))
    )

    candidate = None
    for i_attempt in range(max_attempts):
        position = region.get_random_positions(size=1, rng=rng)[0]
        new_particle = copy.deepcopy(particle)
        new_particle = translate_then_rotate(
            new_particle, position=position, use_com=True, rng=rng
        )
        candidate = atoms + new_particle
        if check_atomic_distances(
            candidate,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            excluded_pairs=intra_bond_pairs,
            allow_isolated=False,
        ):
            break
    else:
        ...

    chemical_formula = particle.get_chemical_formula()
    state = "success"
    if candidate is not None:
        if sort_tags:
            candidate = sort_tags_by_species(candidate)
    else:
        state = "failure"

    return candidate, f"insert_{chemical_formula}_{state}"


if __name__ == "__main__":
    ...
