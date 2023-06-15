#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write

from GDPy.core.register import import_all_modules_for_register
from GDPy.builder.randomBuilder import SurfaceBuilder, ClusterBuilder, BulkBuilder

import_all_modules_for_register()

def test_surface():
    """"""
    params = dict(
        composition = {"Cu": 8, "O": 3},
        region = dict(
            method = "lattice",
            origin = [0., 0., 7.5],
            cell = [5.85, 0.0, 0.0, 0.0, 4.40, 0.0, 0.0, 0.0, 6.]
        ),
        covalent_ratio = [0.4, 2.0],
        test_dist_to_slab = False,
        test_too_far = False,
        random_seed = 1112
    )

    builder = SurfaceBuilder(**params)

    substrates = read("../assets/Cu-fcc-s111p22.xyz", ":")

    frames = builder.run(substrates[0], size=5)
    nframes = len(frames)

    assert nframes == 5


def test_cluster():
    """Test ClusterBuilder with a mixture of atoms and molecules."""
    params = dict(
        composition = {"Cu": 13, "H2O": 3},
        cell = (np.eye(3)*30.).tolist(),
        region = dict(
            method = "lattice",
            origin = [10., 10., 10.],
            cell = (10.*np.eye(3)).flatten()
        ),
        covalent_ratio = [0.6, 2.0],
        test_too_far = False,
        random_seed = 1112,
    )

    builder = ClusterBuilder(**params)

    frames = builder.run(size=20)
    nframes = len(frames)

    assert nframes == 20


def test_bulk():
    """"""
    params = dict(
        composition = {"Cu": 4, "O": 2},
        cell_bounds = dict(
            phi = [35, 145],
            chi = [35, 145],
            psi = [35, 145],
            a = [3, 50],
            b = [3, 50],
            c = [3, 50]
        ),
        random_seed = 1112
    )

    builder = BulkBuilder(**params)

    frames = builder.run(size=20)
    nframes = len(frames)

    assert nframes == 20


if __name__ == "__main__":
    ...
