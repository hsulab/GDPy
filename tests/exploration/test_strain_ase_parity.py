"""Parity checks for GDPy's independent ASE-GA strain implementation."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.strain import StrainMutation
from gdpx.structures.geometry.ga import CellBounds, closest_distances_generator

ase_standard_mutations = pytest.importorskip("ase_ga.standardmutations")
ase_utilities = pytest.importorskip("ase_ga.utilities")
ASEStrainMutation = ase_standard_mutations.StrainMutation


@pytest.mark.parametrize("use_tags", [False, True])
def test_strain_matches_ase_ga_for_identical_generator_state(use_tags):
    parent = Atoms(
        "Cu4",
        scaled_positions=[[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        tags=[1, 1, 2, 2],
    )
    minimum_distances = closest_distances_generator(parent.numbers, 0.5)
    bounds = {"a": [4.0, 12.0], "b": [4.0, 12.0], "c": [4.0, 12.0]}
    common = dict(
        blmin=minimum_distances,
        stddev=0.2,
        number_of_variable_cell_vectors=3,
        use_tags=use_tags,
    )
    gdpx_mutation = StrainMutation(
        **common,
        cellbounds=CellBounds(bounds),
        rng=np.random.default_rng(42),
    )
    ase_mutation = ASEStrainMutation(
        **common,
        cellbounds=ase_utilities.CellBounds(bounds),
        rng=np.random.default_rng(42),
    )

    gdpx_child = gdpx_mutation.mutate(parent.copy())
    ase_child = ase_mutation.mutate(parent.copy())

    assert gdpx_child is not None
    assert ase_child is not None
    np.testing.assert_allclose(gdpx_child.cell, ase_child.cell, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
