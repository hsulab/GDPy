#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

from ase import Atoms


def get_tags_per_species(atoms: Atoms) -> dict[str, list[tuple[int, list[int]]]]:
    """Get tags per species.

    Args:
        atoms: An Atoms object with tags.

    Returns:
        A dict with chemical_formula as the key and the nested dict with
        atom indices to form the molecule.

    Example:

        .. code-block:: python

            >>> atoms = Atoms("PtPtPtCOCO")
            >>> tags = [0, 0, 0, 1, 1, 2, 2]
            >>> atoms.set_tags(tags)
            >>> get_tags_per_species(atoms)
            >>> {'Pt3': [(0, [0,1,2])], 'CO': [(1, [3,4]), (2, [5,6])]}

    """
    # Get tags which is all zero for default
    tags = atoms.get_tags()

    # Group atoms by tags
    tags_dict = {}
    for key, group in itertools.groupby(enumerate(tags), key=lambda x: x[1]):
        atomic_indices = [x[0] for x in group]
        entity = atoms[atomic_indices]  # This gives an Atoms object
        formula = entity.get_chemical_formula()  # type: ignore
        if formula not in tags_dict:
            tags_dict[formula] = []
        tags_dict[formula].append((key, atomic_indices))

    return tags_dict


def reassign_tags_by_species(atoms: Atoms) -> Atoms:
    """"""
    tags_dict = get_tags_per_species(atoms)

    # Find substrate which has tag 0
    substrate: str = ""
    num_atoms_in_substrate: int = 0
    for k, v in tags_dict.items():
        num_instances = len(v)
        v_ = sorted(v, key=lambda x: x[0])  # Make sure we have the entry that has tag=0 at the first
        if v[0][0] == 0:
            assert num_instances == 1, f"`{atoms}` must have only one substrate (tag==0)."
            substrate = k
            num_atoms_in_substrate = len(v[0][1])  # type: ignore
            break
    else:
        tag_min = atoms.get_tags().min()
        assert tag_min > 0, f"`{atoms}` must have tags greater than 0 if no substrate (tag==0) is found."

    new_tags = [0] * num_atoms_in_substrate
    new_indices = list(range(num_atoms_in_substrate))

    current_tag = 1
    valid_keys = sorted([k for k in tags_dict.keys() if k != substrate])
    for species in valid_keys:
        for k, v in tags_dict[species]:  # type: ignore
            new_indices.extend(v)
            new_tags.extend([current_tag] * len(v))
            current_tag += 1

    new_atoms: Atoms = atoms[new_indices]  # type: ignore
    new_atoms.set_tags(new_tags)

    # Inherit info
    new_atoms.info = copy.deepcopy(atoms.info)

    return new_atoms


if __name__ == "__main__":
    ...
