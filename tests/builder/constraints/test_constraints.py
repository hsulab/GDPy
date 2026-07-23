#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from ase import Atoms

from gdpx.group.constraint import canonicalise_constraint_expression, evaluate_constraint_expression

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

def test_canonicalise_constraint_expression():
    assert canonicalise_constraint_expression("1:2") == "`id 1:2`"

def test_evaluate_constraint_expression(H2):
    mobile, frozen = evaluate_constraint_expression(H2, "1:2")
    assert mobile == []
    assert frozen == [0, 1]

if __name__ == "__main__":
    ...
