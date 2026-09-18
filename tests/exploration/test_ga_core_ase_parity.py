"""Parity checks for shared ASE-GA-compatible operator primitives."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.core import OffspringCreator, OperationSelector

ase_offspring = pytest.importorskip("ase.ga.offspring_creator")


def test_operation_selector_matches_ase_ga():
    operations = [object(), object(), object()]
    gdpx = OperationSelector([1.0, 2.0, 3.0], operations, rng=np.random.default_rng(42))
    ase = ase_offspring.OperationSelector([1.0, 2.0, 3.0], operations, rng=np.random.default_rng(42))
    assert [gdpx.get_operator() for _ in range(20)] == [ase.get_operator() for _ in range(20)]


def test_offspring_initialization_matches_ase_ga():
    parent = Atoms("Cu", cell=[5, 5, 5], pbc=[True, False, True])
    child = Atoms("Ni", positions=[[1, 2, 3]])
    gdpx = OffspringCreator.initialize_individual(parent, child)
    ase = ase_offspring.OffspringCreator.initialize_individual(parent, child)
    assert gdpx == ase
    assert gdpx.info == ase.info
