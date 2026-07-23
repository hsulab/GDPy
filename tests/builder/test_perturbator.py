#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from ase.io import read, write

from gdpx.builder.species import MoleculeBuilder
from gdpx.builder.perturbator import PerturbatorBuilder


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


if __name__ == "__main__":
    ...
