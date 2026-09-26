"""Parity checks for GDPy's independent ASE-GA soft implementation."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.soft import SoftMutation
from gdpx.structures.geometry.ga import closest_distances_generator

ase_soft_mutation = pytest.importorskip("ase_ga.soft_mutation")
ASESoftMutation = ase_soft_mutation.SoftMutation


@pytest.mark.parametrize("use_tags", [False, True])
def test_soft_matches_ase_ga(use_tags):
    parent = Atoms(
        "Cu4",
        positions=[[2.0, 2.0, 2.0], [4.4, 2.0, 2.0], [2.0, 4.4, 2.0], [4.4, 4.4, 2.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
        tags=[1, 1, 2, 2],
        info={"confid": 7},
    )
    minimum_distances = closest_distances_generator(parent.numbers, 0.5)
    parameters = dict(
        blmin=minimum_distances,
        bounds=(0.1, 0.5),
        rcut=6.0,
        used_modes_file=None,
        use_tags=use_tags,
    )
    gdpx_child = SoftMutation(**parameters).mutate(parent.copy())
    ase_child = ASESoftMutation(**parameters).mutate(parent.copy())

    assert gdpx_child is not None
    assert ase_child is not None
    np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
