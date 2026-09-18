"""Seeded parity checks for GDPy's ASE-GA-compatible crossovers."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.crossover import (
    ClusterCutAndSpliceCrossover,
    PeriodicCutAndSpliceCrossover,
)
from gdpx.structures.geometry.ga import closest_distances_generator

ase_crossovers = pytest.importorskip("ase_ga.particle_crossovers")
ase_pairing = pytest.importorskip("ase_ga.cutandsplicepairing")


class GeneratorAdapter:
    """Expose both legacy ``randint`` and Generator ``integers`` APIs."""

    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)

    def randint(self, *args, **kwargs):
        return self.generator.integers(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.generator, name)


def _cluster_parents():
    first = Atoms(
        "Cu2Ni2",
        positions=[[-2, -1, 0], [-2, 1, 0], [2, -1, 0], [2, 1, 0]],
        info={"confid": 1},
    )
    second = Atoms(
        "Cu2Ni2",
        positions=[[-1, -2, 0], [1, -2, 0], [-1, 2, 0], [1, 2, 0]],
        info={"confid": 2},
    )
    return first, second


@pytest.mark.parametrize("keep_composition", [False, True])
def test_cluster_cut_splice_matches_ase_ga(keep_composition):
    first, second = _cluster_parents()
    minimum_distances = closest_distances_generator(first.numbers, 0.5)
    gdpx = ClusterCutAndSpliceCrossover(
        minimum_distances,
        keep_composition=keep_composition,
        rng=np.random.default_rng(42),
    )
    ase = ase_crossovers.CutSpliceCrossover(
        minimum_distances,
        keep_composition=keep_composition,
        rng=np.random.default_rng(42),
    )

    gdpx_child, gdpx_description = gdpx.get_new_individual([first.copy(), second.copy()])
    ase_child, ase_description = ase.get_new_individual([first.copy(), second.copy()])

    assert gdpx_description == ase_description.replace(
        "CutSpliceCrossover", "ClusterCutAndSpliceCrossover"
    )
    np.testing.assert_array_equal(gdpx_child.numbers, ase_child.numbers)
    np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
    np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_tags", [False, True])
@pytest.mark.parametrize("seed", [1, 42])
def test_periodic_cut_splice_matches_ase_ga(use_tags, seed):
    slab = Atoms(cell=[12, 12, 12], pbc=False)
    first = Atoms(
        "Cu4",
        positions=[[2, 2, 2], [4, 2, 2], [2, 4, 2], [4, 4, 2]],
        cell=slab.cell,
        pbc=False,
        tags=[1, 2, 3, 4],
        info={"confid": 1},
    )
    second = Atoms(
        "Cu4",
        positions=[[2, 2, 3], [4, 2, 3], [2, 4, 3], [4, 4, 3]],
        cell=slab.cell,
        pbc=False,
        tags=[1, 2, 3, 4],
        info={"confid": 2},
    )
    minimum_distances = closest_distances_generator(first.numbers, 0.5)
    parameters = dict(slab=slab, n_top=4, blmin=minimum_distances, use_tags=use_tags)
    gdpx = PeriodicCutAndSpliceCrossover(**parameters, rng=GeneratorAdapter(seed))
    ase = ase_pairing.CutAndSplicePairing(**parameters, rng=GeneratorAdapter(seed))

    gdpx_child, gdpx_description = gdpx.get_new_individual([first.copy(), second.copy()])
    ase_child, ase_description = ase.get_new_individual([first.copy(), second.copy()])

    assert gdpx_description == ase_description
    assert (gdpx_child is None) == (ase_child is None)
    if gdpx_child is not None:
        np.testing.assert_array_equal(gdpx_child.numbers, ase_child.numbers)
        np.testing.assert_array_equal(gdpx_child.get_tags(), ase_child.get_tags())
        np.testing.assert_allclose(gdpx_child.positions, ase_child.positions, atol=0.0, rtol=0.0)
