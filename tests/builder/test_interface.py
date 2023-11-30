#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import pytest
import tempfile

from ase import Atoms
from ase.io import read, write

from gdpx.core.register import import_all_modules_for_register
from gdpx.builder.interface import build_structures

import_all_modules_for_register()

""""""

@pytest.fixture
def builder_config():
    """"""
    params = dict(
        method = "random_surface",
        composition = {"H2O": "density 0.998"},
        region = dict(
            method = "lattice",
            origin = [0., 0., 7.5],
            cell = [5.08, 0.0, 0.0, 0.0, 4.40, 0.0, 0.0, 0.0, 6.]
        ),
        substrates = "../assets/Cu-fcc-s111p22.xyz",
        covalent_ratio = [0.6, 2.0],
        test_dist_to_slab = False,
        test_too_far = False,
        random_seed = 1112
    )

    return params

@pytest.fixture
def builder_substrate():
    """"""
    atoms = Atoms(
        "Cu16", positions = [
            [-0.00000000,       1.46813179,       0.00000000],
            [-0.00000000,       2.93626357,       2.07625188],
            [ 0.00000000,       0.00000000,       4.15250376],
            [-0.00000000,       1.46813179,       6.22875565],
            [ 1.27143942,       3.67032946,       0.00000000],
            [ 1.27143942,       0.73406589,       2.07625188],
            [ 1.27143942,       3.67032946,       6.22875565],
            [ 1.27143942,       2.20219768,       4.15250376],
            [ 2.54287884,       1.46813179,       0.00000000],
            [ 2.54287884,       2.93626357,       2.07625188],
            [ 2.54287884,       0.00000000,       4.15250376],
            [ 2.54287884,       1.46813179,       6.22875565],
            [ 3.81431827,       3.67032946,       0.00000000],
            [ 3.81431827,       0.73406589,       2.07625188],
            [ 3.81431827,       3.67032946,       6.22875565],
            [ 3.81431827,       2.20219768,       4.15250376]
        ],
        cell = [
            [5.06, 0.0,  0.0], 
            [0.00, 4.4,  0.0],
            [0.00, 0.0, 30.4]
        ],
        pbc = True
    )

    return atoms


def test_interface(builder_substrate, builder_config):
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        #write(tmpdir/"Cu-fcc-s111p22.xyz", builder_substrate)
        _ = build_structures(builder_config, size=5, directory=tmpdir)

        frames = read(tmpdir/"structures.xyz", ":")
        nframes = len(frames)
    
    assert nframes == 5


if __name__ == "__main__":
    ...
