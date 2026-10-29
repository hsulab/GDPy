"""Region extinction selects individual atoms independently of fragment tags."""
import copy

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.persist.thanos import dispatch_thanos
from gdpx.exploration.genetic_algorithm.generation import extinct_candidate


REGION = dict(method="cube", origin=[0, 0, 0], boundary=[-100, -100, -100, 100, 100, 5.5])


@pytest.mark.parametrize("z, expected", [(4.5, 1), (5.5, 1), (5.5001, 0), (8, 0)])
def test_carbon_region(z, expected):
    atoms = Atoms("PdC2", positions=[[0, 0, 2], [0, 0, 8], [2, 0, z]])
    original = atoms.positions.copy()
    callback = dispatch_thanos(name="region", group="`symbol C`", region=REGION)
    assert callback(atoms) == expected
    np.testing.assert_array_equal(atoms.positions, original)
    atoms.info["key_value_pairs"] = {}
    extinct_candidate(atoms, [callback])
    assert atoms.info["key_value_pairs"]["extinct"] == expected


def test_selection_and_reuse_without_implicit_wrapping():
    atoms = Atoms("PdC", positions=[[0, 0, 2], [0, 0, 24]], cell=[10, 10, 20], pbc=True)
    callback = dispatch_thanos(name="region", group="`symbol C`", region=REGION)
    assert callback(atoms) == 0
    assert atoms.positions[1, 2] == 24
    atoms.pbc = False
    assert callback(atoms) == 0
    assert callback(atoms[:1]) == 0
    atoms.positions[1, 2] = 4
    assert callback(atoms) == 1
    assert dispatch_thanos(name="region", group="`symbol C` and `index 0`", region=REGION)(atoms) == 0


@pytest.mark.parametrize("kwargs", [{"region": REGION}, {"group": "`symbol C`"}])
def test_required_configuration(kwargs):
    with pytest.raises(ValueError):
        dispatch_thanos(name="region", **kwargs)


def test_region_configuration_is_not_mutated():
    rule = dict(name="region", group="`symbol C`", region=copy.deepcopy(REGION))
    original = copy.deepcopy(rule)
    callback = dispatch_thanos(**rule)
    assert rule == original
    assert callback(Atoms("C", positions=[[0, 0, 4]])) == 1
    assert callback(Atoms("C", positions=[[0, 0, 7]])) == 0


@pytest.mark.parametrize("wrap, expected", [(True, 1), ([-0.05, 0.95], 1),
                                           ([0, 1], 0), ([-0.5, 0.5], 1), (False, 0)])
def test_fractional_wrapping_interval(wrap, expected):
    atoms = Atoms("O", positions=[[3, 4, 19.5]], cell=[10, 10, 20], pbc=True)
    original = atoms.positions.copy()
    callback = dispatch_thanos(name="region", group="`symbol O`",
                               region=dict(method="surface", z_range=[-1, 0], wrap=wrap))
    assert callback(atoms) == expected
    np.testing.assert_array_equal(atoms.positions, original)


def test_default_wrap_and_half_open_interval():
    # Default maps both endpoints to -0.1 and handles multiple cell crossings.
    atoms = Atoms("O4", positions=[[0, 0, z] for z in [-2, 18, 58, -42]],
                  cell=[10, 10, 20], pbc=True)
    callback = dispatch_thanos(name="region", group="`symbol O`",
                               region=dict(method="surface", z_range=[-2.001, -1.999]))
    for atom in atoms:
        assert callback(Atoms("O", positions=[atom.position], cell=atoms.cell, pbc=True)) == 1


def test_custom_wrap_respects_pbc_and_skew_cell():
    atoms = Atoms("O", scaled_positions=[[0.2, 0.3, 0.98]],
                  cell=[[10, 0, 0], [2, 10, 0], [1, 2, 20]], pbc=[True, True, False])
    callback = dispatch_thanos(name="region", group="`symbol O`",
                               region=dict(method="surface", z_range=[-1, 0], wrap=[-0.2, 0.8]))
    assert callback(atoms) == 0
    atoms.pbc[2] = True
    assert callback(atoms) == 1


@pytest.mark.parametrize("wrap", [[0, 0.9], [1, 0], [0, 2], [0], [0, 1, 2],
                                  [float("nan"), 1], [0, float("inf")], [False, 1], "0 1", 1])
def test_invalid_wrap_interval(wrap):
    with pytest.raises(ValueError, match="wrap"):
        dispatch_thanos(name="region", group="`symbol O`",
                        region=dict(method="surface", z_range=[-1, 0], wrap=wrap))


def test_wrap_configuration_is_copied():
    wrap = [0, 1]
    callback = dispatch_thanos(name="region", group="`symbol O`",
                               region=dict(method="surface", z_range=[-1, 0], wrap=wrap))
    wrap[:] = [-0.05, 0.95]
    assert callback(Atoms("O", positions=[[0, 0, 19.5]], cell=[10, 10, 20], pbc=True)) == 0
