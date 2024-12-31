#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


if __name__ == "__main__":
    ...
