#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from ase.io import write

from GDPy.builder.species import MoleculeBuilder
from GDPy.builder.packer import PackerBuilder


def test_packer():
    """"""
    water = MoleculeBuilder(name="H2O").run()[0]
    builder = PackerBuilder(
        substrates=[water], numbers=[4], intermoleculer_distance=[1., 8.],
        random_seed=1112
    )
    structures = builder.run(size=5)
    n_structures = len(structures)

    #write("./xxx.xyz", structures)

    assert n_structures == 5


def test_packer_mixed():
    """"""
    water = MoleculeBuilder(name="H2O").run()[0]
    methanol = MoleculeBuilder(name="CH3OH").run()[0]
    builder = PackerBuilder(
        substrates=[water, methanol], numbers=[4,2], 
        intermoleculer_distance=[1., 10.], random_seed=1112
    )
    structures = builder.run(size=5)
    n_structures = len(structures)

    #write("./xxx.xyz", structures)

    assert n_structures == 5


if __name__ == "__main__":
    ...