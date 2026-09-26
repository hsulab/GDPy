"""Numerical parity checks for GDPy's ASE-GA-compatible comparators."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.population.comparators.basic import NNMatComparator
from gdpx.exploration.population.comparators.ofp import OFPComparator

ase_ofp = pytest.importorskip("ase_ga.ofp_comparator")
ase_particle = pytest.importorskip("ase_ga.particle_comparator")


def test_nnmat_matches_ase_ga():
    first = Atoms(
        "Cu3Ni",
        positions=[[2, 2, 2], [4.4, 2, 2], [2, 4.4, 2], [4.4, 4.4, 2]],
        cell=[20, 20, 20],
        pbc=False,
    )
    second = first.copy()
    second.positions[3, 0] += 0.3
    parameters = dict(d=0.2, elements=["Cu", "Ni"], mic=False)
    assert NNMatComparator(**parameters).looks_like(first, second) == (
        ase_particle.NNMatComparator(**parameters).looks_like(first, second)
    )


@pytest.mark.parametrize("pbc", [False, True])
def test_ofp_distance_matches_ase_ga(pbc):
    first = Atoms(
        "Cu3Ni",
        positions=[[2, 2, 2], [4.4, 2, 2], [2, 4.4, 2], [4.4, 4.4, 2]],
        cell=[12, 12, 12],
        pbc=pbc,
    )
    second = first.copy()
    second.positions[3] += [0.2, -0.1, 0.1]
    parameters = dict(rcut=5.0, binwidth=0.1, sigma=0.05, nsigma=4, pbc=pbc)
    gdpx_distance = OFPComparator(**parameters)._compare_structure(first.copy(), second.copy())
    ase_distance = ase_ofp.OFPComparator(**parameters)._compare_structure(first.copy(), second.copy())
    np.testing.assert_allclose(gdpx_distance, ase_distance, atol=1e-15, rtol=1e-12)
