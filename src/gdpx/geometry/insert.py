#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .spatial import check_atomic_distances


def translate_then_rotate(atoms, position, use_com: bool, rng):
    """"""
    num_atoms = len(atoms)
    if not use_com:
        center = np.mean(atoms.positions, axis=0)
    else:
        center = atoms.get_center_of_mass()

    # Translate
    atoms.translate(position - center)

    # Rotate
    if num_atoms > 1:
        phi, theta, psi = 360 * rng.uniform(0, 1, 3)
        atoms.euler_rotate(phi=phi, theta=0.5 * theta, psi=psi, center=position)

    return atoms


def insert_fragments_at_once(
    substrate: Atoms,
    fragments: List[Atoms],
    region,
    molecular_distances,
    covalent_ratio,
    bond_distance_dict,
    random_state,
) -> Optional[Atoms]:
    """"""
    # Initialise a random number generator
    rng = np.random.Generator(np.random.PCG64(random_state))

    # Overwrite tags for atoms in the substrate
    atoms = substrate
    # atoms.set_tags(max(atoms.get_tags()) + 1)
    atoms.set_tags(0)

    # Sort fragments by chemical formulae alphabetically
    fragments = sorted(fragments, key=lambda a: a.get_chemical_formula())
    num_fragments = len(fragments)

    # Find intra-molecular pairs
    excluded_pairs = list(itertools.permutations(range(len(atoms)), 2))

    end_indices = np.cumsum([len(a) for a in fragments])
    beg_indices = np.hstack([[0], end_indices[:-1]])

    intra_bonds = itertools.chain(
        *[
            itertools.permutations(range(beg, end), 2)
            for beg, end in zip(beg_indices, end_indices)
        ]
    )
    excluded_pairs.extend(intra_bonds)

    # Check inter-molecular distances
    min_molecular_distance, max_molecular_distance = molecular_distances
    random_positions = region.get_random_positions(size=num_fragments, rng=rng)
    if num_fragments > 1:
        pair_positions = np.array(list(itertools.combinations(random_positions, 2)))
        raw_vectors = pair_positions[:, 0, :] - pair_positions[:, 1, :]
        mic_vecs, mic_dis = find_mic(v=raw_vectors, cell=atoms.cell, pbc=atoms.pbc)
        if (
            np.min(mic_dis) >= min_molecular_distance
            and np.max(mic_dis) <= max_molecular_distance
        ):
            is_molecule_valid = True
        else:
            is_molecule_valid = False
    else:
        is_molecule_valid = True

    if is_molecule_valid:
        for a, p in zip(fragments, random_positions):
            # rotate and translate
            a = copy.deepcopy(a)
            a = translate_then_rotate(a, position=p, use_com=True, rng=rng)
            a.set_tags(int(np.max(atoms.get_tags()) + 1))
            atoms += a

        if not check_atomic_distances(
            atoms,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            excluded_pairs=excluded_pairs,
            allow_isolated=False,
        ):
            atoms = None
        else:
            ...
    else:
        atoms = None

    return atoms

def batch_insert_fragments_at_once(
    substrates: List[Atoms],
    fragments,
    region,
    molecular_distances,
    covalent_ratio,
    bond_distance_dict,
    random_states: List[int],
):
    """"""
    frames = []
    for atoms, random_state in zip(substrates, random_states):
        new_atoms = insert_fragments_at_once(
            atoms,
            fragments,
            region,
            molecular_distances,
            covalent_ratio,
            bond_distance_dict,
            random_state=random_state,
        )
        frames.append(new_atoms)

    return frames



if __name__ == "__main__":
    ...
