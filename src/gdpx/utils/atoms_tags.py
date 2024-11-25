#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import List, Mapping

from ase import Atoms


def get_tags_per_species(atoms: Atoms) -> Mapping[str, Mapping[int, List[int]]]:
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
            >>> {'Pt3': {0: [0,1,2]}, 'CO': {1: [3,4], 2: [5,6]}}

    """

    tags = atoms.get_tags()  # default is all zero

    tags_dict = {}  # species -> tag list
    for key, group in itertools.groupby(enumerate(tags), key=lambda x: x[1]):
        cur_indices = [x[0] for x in group]
        # print(key, " :", cur_indices)
        cur_atoms = atoms[cur_indices]
        formula = cur_atoms.get_chemical_formula()
        # print(formula)
        # print(key)
        if formula not in tags_dict:
            tags_dict[formula] = []
        tags_dict[formula].append([key, cur_indices])

    return tags_dict


if __name__ == "__main__":
    ...
