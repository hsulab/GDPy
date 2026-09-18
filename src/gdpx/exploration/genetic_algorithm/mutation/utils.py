"""Shared helpers for genetic mutations."""

import numpy as np
from ase import Atoms


def movable_groups(atoms: Atoms, n_top: int, use_tags: bool) -> list[np.ndarray]:
    """Return independently movable atom or tagged-fragment indices."""
    first = len(atoms) - n_top
    if not use_tags:
        return [np.array([index], dtype=int) for index in range(first, len(atoms))]
    tags = atoms.get_tags()
    return [np.flatnonzero(tags == tag) for tag in np.unique(tags[first:]) if tag != 0]
