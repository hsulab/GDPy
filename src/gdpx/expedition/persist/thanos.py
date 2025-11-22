#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
import itertools
from typing import Callable, Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from gdpx.utils.atoms_tags import get_tags_per_species


def extinct_by_small_pair_distance(atoms: Atoms, pair_distance_dict: dict[tuple[str, str], float]) -> Literal[0, 1]:
    """"""
    box = atoms.get_cell()
    pbc = atoms.get_pbc()

    identities = get_tags_per_species(atoms)

    extinct = 0
    for (s_i, s_j), dmin in pair_distance_dict.items():
        if s_i not in identities or s_j not in identities:
            continue

        for (_, ind_i), (_, ind_j) in itertools.product(identities[s_i], identities[s_j]):
            raw_vectors = np.reshape(
                atoms.positions[ind_i][:, np.newaxis, :] - atoms.positions[ind_j][np.newaxis, :, :], (-1, 3)
            )
            _, dists = find_mic(raw_vectors, box, pbc)
            if np.min(dists) < dmin:
                extinct = 1

    return extinct


def thanos_by_pair_distance(
    pair_distance_min: list[tuple[str, str, float]],
) -> Callable[[Atoms], Literal[0, 1]]:
    """"""
    pair_distance_dict = {}
    for s_i, s_j, dmin in pair_distance_min:
        pair_distance_dict[(s_i, s_j)] = dmin

    return functools.partial(extinct_by_small_pair_distance, pair_distance_dict=pair_distance_dict)


THANOS_CALLBACKS = dict(
    pair_distance=thanos_by_pair_distance,
)


if __name__ == "__main__":
    ...
