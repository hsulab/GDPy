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


def test_wrapping_selection_and_reuse():
    atoms = Atoms("PdC", positions=[[0, 0, 2], [0, 0, 24]], cell=[10, 10, 20], pbc=True)
    callback = dispatch_thanos(name="region", group="`symbol C`", region=REGION)
    assert callback(atoms) == 1
    assert atoms.positions[1, 2] == 24
    assert dispatch_thanos(name="region", group="`symbol C`", region=REGION, wrap=False)(atoms) == 0
    atoms.pbc = False
    assert callback(atoms) == 0
    assert callback(atoms[:1]) == 0
    atoms.positions[1, 2] = 4
    assert callback(atoms) == 1
    assert dispatch_thanos(name="region", group="`symbol C` and `index 0`", region=REGION)(atoms) == 0


@pytest.mark.parametrize("kwargs", [{"region": REGION}, {"group": "`symbol C`"},
                                      {"region": REGION, "group": "`symbol C`", "wrap": "false"}])
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
