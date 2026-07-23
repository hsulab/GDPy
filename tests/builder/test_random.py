#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write

from gdpx.builder.random_bulk import RandomBulkBuilder
from gdpx.builder.utils import compute_molecule_number_from_density

def test_number():
    """"""

    number = compute_molecule_number_from_density(18, 20*14*11, 0.998)

    assert number == 102


def test_bulk():
    """"""
    params = dict(
        composition = {"Cu": 4, "O": 2},
        box=dict(bounds=dict(phi=[35, 145], chi=[35, 145], psi=[35, 145], a=[3, 50], b=[3, 50], c=[3, 50])),
        random_seed = 1112
    )

    builder = RandomBulkBuilder(**params)

    frames = builder.run(size=20)
    nframes = len(frames)

    assert nframes == 20


if __name__ == "__main__":
    ...
