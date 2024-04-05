#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from ase import Atoms

from gdpx.builder.constraints import convert_indices, parse_constraint_info

@pytest.fixture(scope="function")
def rng():
    """Initialise a random number generator."""
    rng = np.random.default_rng(seed=1112)

    return rng

@pytest.fixture(autouse=True)
def H2():
    """Create a H2 molecule."""
    atoms = Atoms(
        symbols="H2", positions=[[0., 0., 0.], [0., 0., 1.]],
        cell=np.eye(3)*10.
    )

    return atoms

def test_convert_indices():
    """"""
    ret = convert_indices([1,2,3,6,7,8], index_convention="lmp")

    assert ret == "1:3 6:8"

def test_parse_constraint_info(H2):
    """"""
    mobile_text, frozen_text = parse_constraint_info(
        atoms=H2, cons_text="1:2", ignore_ase_constraints=True, ret_text=True
    )

    assert mobile_text == ""
    assert frozen_text == "1:2"

if __name__ == "__main__":
    ...
