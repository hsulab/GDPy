"""Parity checks for GDPy's independent ASE-GA rattle implementation."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.rattle import RattleMutation
from gdpx.structures.geometry.ga import closest_distances_generator

ase_standard_mutations = pytest.importorskip("ase_ga.standardmutations")
ASERattleMutation = ase_standard_mutations.RattleMutation


@pytest.mark.parametrize("use_tags", [False, True])
def test_rattle_matches_ase_ga_for_identical_generator_state(use_tags):
    parent = Atoms(
        "Cu5",
        positions=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.5, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 1.5, 0.0],
        ],
        cell=[15.0, 15.0, 15.0],
        pbc=False,
        tags=[0, 1, 1, 2, 2],
    )
    minimum_distances = closest_distances_generator(parent.numbers, 0.5)
    parameters = dict(
        blmin=minimum_distances,
        n_top=4,
        rattle_strength=0.4,
        rattle_prop=0.75,
        test_dist_to_slab=True,
        use_tags=use_tags,
    )
    gdpx_mutation = RattleMutation(**parameters, rng=np.random.default_rng(42))
    ase_mutation = ASERattleMutation(**parameters, rng=np.random.default_rng(42))

    gdpx_child = gdpx_mutation.mutate(parent)
    ase_child = ase_mutation.mutate(parent)

    assert gdpx_child is not None
    assert ase_child is not None
    assert gdpx_child == ase_child
    np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
    np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=0.0, rtol=0.0)
