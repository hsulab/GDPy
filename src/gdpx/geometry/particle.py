#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase import Atoms


def translate_then_rotate(atoms: Atoms, position, use_com: bool, rng: np.random.Generator):
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


if __name__ == "__main__":
    ...
