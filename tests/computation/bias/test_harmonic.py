#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import pytest

import numpy as np

from ase.io import read, write


@pytest.fixture
def water():
    """"""
    xyz_fpath = pathlib.Path(__file__).resolve().parent.parent/"assets"/"H2O.xyz"
    atoms = read(xyz_fpath)

    return atoms


def test_harmonic(water):
    """"""
    atoms = water
    print(atoms.positions)

    colvar = dict(
        name = "position",
        axis = 2
    )

    calc = HarmonicBias(colvar=colvar, k=100., s=0.2)
    atoms.calc = calc

    forces = atoms.get_forces()

    assert np.allclose(
        forces, 
        [
            [0., 0., -19.6309], 
            [0., 0., 99.630905], 
            [0., 0., 99.630905]
        ]
    )


if __name__ == "__main__":
    ...
