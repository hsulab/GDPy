#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from ase.io import read, write

from gdpx.builder.species import MoleculeBuilder
from gdpx.builder.perturbator import PerturbatorBuilder
from gdpx.builder.insert import InsertModifier


@pytest.mark.basic
def test_molecule():
    """"""
    inp = MoleculeBuilder(name="H2O").run()

    builder = PerturbatorBuilder(eps=0.2, ceps=0.02, random_seed=1112)
    structures = builder.run(substrates=inp, size=10)
    n_structures = len(structures)

    assert n_structures == 10


@pytest.mark.basic
def test_cluster():
    """"""
    substrates = read("./assets/Pd38.xyz", ":")

    builder = PerturbatorBuilder(eps=0.2, ceps=None, random_seed=1112)
    structures = builder.run(substrates=substrates, size=10)
    n_structures = len(structures)

    assert n_structures == 10


@pytest.mark.basic
def test_insert():
    """"""
    substrates = read("./assets/Pd38.xyz", ":")

    builder = InsertModifier(
        # region = dict(method="sphere", origin=[12., 12., 12.], radius=10),
        region=dict(
            method="intersect",
            regions=[
                dict(method="sphere", origin=[12.0, 12.0, 12.0], radius=8.0),
                dict(method="sphere", origin=[12.0, 12.0, 12.0], radius=6.0),
            ],
        ),
        composition=dict(O=4),
        max_times_size=100,
        random_seed=1112,
    )
    structures = builder.run(substrates=substrates, size=10)
    n_structures = len(structures)

    # write("./xxx.xyz", structures)

    assert n_structures == 10


if __name__ == "__main__":
    ...
