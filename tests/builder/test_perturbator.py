#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from GDPy.builder.species import MoleculeBuilder
from GDPy.builder.perturbator import PerturbatorBuilder


def test_molecule():
    """"""
    inp = MoleculeBuilder(name="H2O").run()

    builder = PerturbatorBuilder(eps=0.2)
    structures = builder.run(substrates=inp, size=10)
    n_structures = len(structures)

    assert n_structures == 10


if __name__ == "__main__":
    ...