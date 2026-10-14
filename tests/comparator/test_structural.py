import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.analysis.comparators import create_comparator
from gdpx.analysis.comparators.oganov_fingerprint import OganovFingerprintComparator
from gdpx.analysis.comparators.displacement import DisplacementComparator
from gdpx.exploration.population.comparators import create_population_comparator


def cluster():
    return Atoms("Cu3Ni", positions=[[2, 2, 2], [4.4, 2, 2], [2, 4.4, 2], [4.4, 4.4, 2]],
                 cell=[12, 12, 12], tags=[1, 1, 1, 1])


@pytest.mark.parametrize("pbc,expected", [(False, 0.11578741807378512), (True, 0.12005598650129512)])
def test_oganov_numerical_regression(pbc, expected):
    first = cluster()
    first.pbc = pbc
    second = first.copy()
    second.positions[3] += [0.2, -0.1, 0.1]
    comparator = OganovFingerprintComparator(rcut=5, binwidth=0.1, sigma=0.05, pbc=pbc, random_seed=0)
    distance = comparator.fingerprint_distance(*comparator.prepare_data([first, second]))
    assert distance == pytest.approx(expected, abs=1e-15, rel=1e-12)


@pytest.mark.parametrize("method,cls", [("oganov_fingerprint", OganovFingerprintComparator),
                                        ("displacement", DisplacementComparator)])
def test_factory_and_instance_identity(method, cls):
    parameters = dict(method=method, random_seed=0)
    saved = copy.deepcopy(parameters)
    comparator = create_comparator(parameters)
    assert isinstance(comparator, cls)
    assert create_comparator(comparator) is comparator
    assert parameters == saved
    assert comparator(cluster(), cluster())


@pytest.mark.parametrize("periodic", [False, True])
def test_population_uses_analysis_registration(periodic):
    comparator = create_population_comparator(dict(method="oganov_fingerprint", random_seed=0), periodic=periodic)
    assert type(comparator) is OganovFingerprintComparator
    assert np.all(comparator.pbc == periodic)
    with pytest.raises(ValueError, match="system.periodic"):
        create_population_comparator(dict(method="oganov_fingerprint", pbc=False))


@pytest.mark.parametrize("method", ["oganov_fingerprint", "displacement"])
@pytest.mark.parametrize("expression", ["`id 2:5`", "`symbol Cu Ni`", "`tag 1`"])
def test_group_selection_and_input_ownership(method, expression):
    first = Atoms("Mg", positions=[[8, 0, 0]], cell=[12] * 3, tags=[0]) + cluster()
    first.info["fingerprints"] = {"stale": [1, 2, 3]}
    second = first.copy()
    second.positions[0] += [1, 2, 3]
    original = first.copy()
    info = copy.deepcopy(first.info)
    comparator = create_comparator(dict(method=method, group_expression=expression, random_seed=0))
    prepared = comparator.prepare_data([first, second])
    assert comparator.compare_fingerprints(*prepared)
    assert comparator.fingerprint_distance(*prepared) == pytest.approx(0, abs=1e-12)
    assert first == original
    assert first.info == info
    assert second.info == info


def test_oganov_subset_matches_extracted_cluster():
    atoms = Atoms("Mg", positions=[[8, 0, 0]], cell=[12] * 3) + cluster()
    atoms.positions[-1] += [0.2, -0.1, 0.1]
    parameters = dict(rcut=5, binwidth=0.1, sigma=0.05, pbc=False, random_seed=0)
    grouped = OganovFingerprintComparator(group_expression="`id 2:5`", **parameters)
    full = OganovFingerprintComparator(**parameters)
    assert full.fingerprint_distance(grouped.prepare_data([atoms])[0], full.prepare_data([atoms[1:]])[0]) < 1e-12


@pytest.mark.parametrize("method", ["oganov_fingerprint", "displacement"])
def test_invalid_groups_and_obsolete_parameter(method):
    with pytest.raises(ValueError, match="group_expression"):
        create_comparator(dict(method=method, n_top=4, random_seed=0))
    empty = create_comparator(dict(method=method, group_expression="`symbol Pt`", random_seed=0))
    with pytest.raises(ValueError, match="empty comparison group"):
        empty.prepare_data([cluster()])
    comparator = create_comparator(dict(method=method, random_seed=0))
    with pytest.raises(ValueError, match="identical compositions"):
        comparator.prepare_data([cluster(), cluster()[:-1]])
    first, second = comparator.prepare_data([cluster()])[0], comparator.prepare_data([cluster()[:-1]])[0]
    with pytest.raises(ValueError, match="identical compositions"):
        comparator.compare_fingerprints(first, second)


def test_oganov_ignores_energy_and_rejects_embedded_energy_gate():
    first, second = cluster(), cluster()
    first.calc = SinglePointCalculator(first, energy=0)
    second.calc = SinglePointCalculator(second, energy=2)
    comparator = OganovFingerprintComparator(pbc=False, random_seed=0)
    assert comparator(first, second)
    assert comparator.compare_fingerprints(*comparator.prepare_data([first, second]))
    assert "fingerprints" not in first.info
    assert "fingerprints" not in second.info
    with pytest.raises(ValueError, match="energetic_oganov_fingerprint"):
        OganovFingerprintComparator(dE=1)


def test_displacement_permutation_invariance():
    first = cluster()
    second = first[[3, 1, 0, 2]]
    comparator = DisplacementComparator(permutation_invariant=True, random_seed=0)
    assert comparator.fingerprint_distance(*comparator.prepare_data([first, second])) < 1e-10


def test_old_registry_name_is_removed():
    with pytest.raises(KeyError):
        create_comparator(dict(method="ofp"))
    with pytest.raises(ValueError, match="Unknown population comparator"):
        create_population_comparator(dict(method="ofp"))
