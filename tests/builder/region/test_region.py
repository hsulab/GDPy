#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from ase import Atoms

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


class TestRegion():

    def test_sphere(self, rng):
        """Test sphere region."""
        print("start one...")
        print(rng)
        print(rng.random(3))
        x = "this"
        assert "h" in x

    def test_lattice(self, rng):
        print("start one...")
        print(rng)
        print(rng.random(3))
        x = "this"
        assert "h" in x


if __name__ == "__main__":
    ...