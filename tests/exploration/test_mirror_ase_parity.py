"""Parity checks for GDPy's independent ASE-GA mirror implementation."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.mirror import MirrorMutation
from gdpx.structures.geometry.ga import closest_distances_generator

ase_standard_mutations = pytest.importorskip("ase.ga.standardmutations")
ASEMirrorMutation = ase_standard_mutations.MirrorMutation


@pytest.mark.parametrize("reflect", [False, True])
def test_mirror_matches_ase_ga_for_identical_generator_state(reflect):
    parent = Atoms(
        symbols=["Cu", "Cu", "Ni", "Cu", "Ni"],
        positions=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
        ],
        cell=[15.0, 15.0, 15.0],
        pbc=False,
    )
    minimum_distances = closest_distances_generator(parent.numbers, 0.5)
    parameters = dict(
        blmin=minimum_distances,
        n_top=4,
        reflect=reflect,
    )
    gdpx_mutation = MirrorMutation(
        **parameters,
        use_tags=False,
        rng=np.random.default_rng(42),
    )
    ase_mutation = ASEMirrorMutation(
        **parameters,
        rng=np.random.default_rng(42),
    )

    gdpx_child = gdpx_mutation.mutate(parent)
    ase_child = ase_mutation.mutate(parent)

    assert gdpx_child is not None
    assert ase_child is not None
    assert gdpx_child == ase_child
    np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
    np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=0.0, rtol=0.0)
