import functools
import itertools
from typing import Callable, Literal, Optional

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
            # remove self-interaction
            if ind_i == ind_j:
                continue
            raw_vectors = np.reshape(
                atoms.positions[ind_i][:, np.newaxis, :] - atoms.positions[ind_j][np.newaxis, :, :], (-1, 3)
            )
            _, dists = find_mic(raw_vectors, box, pbc)
            if np.min(dists) < dmin:
                extinct = 1
                break

        if extinct == 1:
            break

    return extinct


def extinct_by_number_of_particles(
    atoms: Atoms,
    particle: str,
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
) -> Literal[0, 1]:
    """"""
    identities = get_tags_per_species(atoms)
    num_particles = len(identities.get(particle, []))

    extinct = 0
    if min_num is not None and num_particles < min_num:
        extinct = 1
    if max_num is not None and num_particles > max_num:
        extinct = 1

    return extinct


def dispatch_thanos(
    name: str,
    pair_distance_min: Optional[list[tuple[str, str, float]]] = None,
    number: tuple[str, Optional[int], Optional[int]] = ("", None, None),
) -> Callable[[Atoms], Literal[0, 1]]:
    """"""
    if pair_distance_min is not None:
        pair_distance_dict = {}
        for s_i, s_j, dmin in pair_distance_min:
            pair_distance_dict[(s_i, s_j)] = dmin
    else:
        pair_distance_dict = {}

    if name == "pair_distance":
        return functools.partial(extinct_by_small_pair_distance, pair_distance_dict=pair_distance_dict)
    elif name == "particle_number":
        particle, min_num, max_num = number
        return functools.partial(extinct_by_number_of_particles, particle=particle, min_num=min_num, max_num=max_num)
    else:
        raise Exception(f"Thanos function '{name}' is not recognised.")
