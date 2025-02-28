#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase import Atoms


def centre_structures(frames: list[Atoms], vacuum_size:float):
    """Make structures at the centre of their boxes.

    The cells will be expanded by the vacuum size.

    """
    for atoms in frames:
        lengths = np.max(atoms.positions, axis=0) - np.min(atoms.positions, axis=0)
        lengths += vacuum_size

        box = np.zeros((3, 3))
        np.fill_diagonal(box, lengths)
        box_center = np.sum(box, axis=0) / 2.0

        atoms.set_cell(box)
        atoms.positions -= atoms.get_center_of_mass() - box_center

    return


if __name__ == "__main__":
    ...
