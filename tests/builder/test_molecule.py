#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from gdpx.builder.species import MoleculeBuilder


def test_molecule():
    """"""
    builder = MoleculeBuilder(name="H2O")
    structures = builder.run()

    water = structures[0]

    assert len(water) == 3


if __name__ == "__main__":
    ...