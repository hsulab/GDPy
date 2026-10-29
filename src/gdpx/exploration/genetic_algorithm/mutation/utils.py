"""Shared helpers for genetic mutations."""

import numpy as np
from ase import Atoms


def movable_groups(atoms: Atoms, use_tags: bool) -> list[np.ndarray]:
    """Return independently movable atom or tagged-fragment indices."""
    mobile = np.flatnonzero(atoms.get_tags() != 0)
    if not use_tags:
        return [np.array([index], dtype=int) for index in mobile]
    tags = atoms.get_tags()
    return [np.flatnonzero(tags == tag) for tag in np.unique(tags[mobile]) if tag != 0]
