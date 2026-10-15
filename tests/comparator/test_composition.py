import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.analysis.comparators import BaseComparator, create_comparator
from gdpx.analysis.comparators.combined import CombinedComparator
from gdpx.analysis.selectors import create_selector
from gdpx.exploration.population.comparators import create_population_comparator


def candidate(energy, distance=2):
    atoms = Atoms("CuPtO", positions=[[0, 0, 0], [distance, 0, 0], [0, 2, 0]], cell=[12] * 3)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((3, 3)))
    return atoms


@pytest.mark.parametrize("difference,similar", [(0.0, True), (0.999, True), (1.0, False), (2.0, False)])
def test_energy_criterion_and_boundary(difference, similar):
    comparator = create_comparator(dict(method="energy", ene_diff_tol=1, random_seed=0))
    first, second = candidate(0), candidate(difference, distance=6)
    assert comparator(first, second) is similar
    assert comparator.compare_fingerprints(*comparator.prepare_data([first, second])) is similar


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf])
def test_energy_threshold_validation(threshold):
    with pytest.raises(ValueError, match="positive and finite"):
        create_comparator(dict(method="energy", ene_diff_tol=threshold, random_seed=0))


def test_energy_requires_available_finite_energy():
    comparator = create_comparator(dict(method="energy", random_seed=0))
    with pytest.raises(ValueError, match="finite potential energies"):
        comparator.prepare_data([candidate(np.nan)])
    with pytest.raises(RuntimeError, match="no calculator"):
        comparator.prepare_data([Atoms("Cu")])


@pytest.mark.parametrize("method", ["combined", "energetic_oganov_fingerprint"])
@pytest.mark.parametrize("energy,distance,similar", [(0.2, 2, True), (2, 2, False), (0.2, 6, False), (2, 6, False)])
def test_combination_direct_and_prepared_agree(method, energy, distance, similar):
    structure = dict(method="oganov_fingerprint", pbc=False)
    parameters = (dict(method="combined", comparators=[dict(method="energy", ene_diff_tol=1), structure])
                  if method == "combined" else dict(method=method, ene_diff_tol=1, pbc=False))
    original = copy.deepcopy(parameters)
    comparator = create_comparator(parameters)
    first, second = candidate(0), candidate(energy, distance)
    assert comparator(first, second) is similar
    prepared = comparator.prepare_data([first, second])
    assert comparator.compare_fingerprints(*prepared) is similar
    assert parameters == original
    assert first.info == second.info == {}


def test_preset_returns_combined_comparator_with_flat_parameters():
    parameters = dict(method="energetic_oganov_fingerprint", ene_diff_tol=0.2,
                      group_expression="`symbol Cu Pt O`", cos_dist_max=0.01,
                      rcut=8.0, binwidth=0.1, sigma=0.2, nsigma=3, pbc=False,
                      maxdims=[12, 12, 12], random_seed=7, n_jobs=2)
    original = copy.deepcopy(parameters)
    comparator = create_comparator(parameters)
    assert type(comparator) is CombinedComparator
    energy, fingerprint = comparator.comparators
    assert energy.ene_diff_tol == 0.2
    assert fingerprint.name == "oganov_fingerprint"
    for name in ("group_expression", "cos_dist_max", "rcut", "binwidth", "sigma", "nsigma", "maxdims"):
        assert getattr(fingerprint, name) == parameters[name]
    assert not fingerprint.pbc.any()
    assert all(child.random_seed == 7 and child.njobs == 2 for child in comparator.comparators)
    assert parameters == original


def test_factory_accepts_a_list_and_rmsd_combination():
    configuration = [dict(method="energy", ene_diff_tol=1), dict(method="displacement")]
    comparator = create_comparator(configuration)
    assert type(comparator) is CombinedComparator
    assert comparator(candidate(0), candidate(0.2))
    assert not comparator(candidate(0), candidate(0.2, 6))
    assert not comparator(candidate(0), candidate(2))
    direct = CombinedComparator(comparators=comparator.comparators, random_seed=0)
    assert direct.comparators[0] is comparator.comparators[0]



def test_direct_composition_short_circuits_geometry():
    class UnreachableComparator(BaseComparator):
        def looks_like(self, first, second):
            raise AssertionError("Energy rejection must short-circuit geometry.")

    comparator = create_comparator(dict(method="combined", comparators=[dict(method="energy"), UnreachableComparator(random_seed=0)]))
    assert not comparator(candidate(0), candidate(2))


def test_raw_comparator_fallback_preserves_energy_and_inputs():
    class RawComparator(BaseComparator):
        def looks_like(self, first, second):
            assert first.get_potential_energy() == 0
            first.positions += 10
            return True

    comparator = create_comparator(dict(method="combined", comparators=[RawComparator(random_seed=0)]))
    first, second = candidate(0), candidate(0)
    original = first.positions.copy()
    assert comparator.compare_fingerprints(*comparator.prepare_data([first, second]))
    np.testing.assert_array_equal(first.positions, original)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("method", ["combined", "energetic_oganov_fingerprint"])
def test_population_injects_periodicity_recursively(periodic, method):
    parameters = (dict(method="combined", comparators=[dict(method="energetic_oganov_fingerprint")])
                  if method == "combined" else dict(method=method))
    original = copy.deepcopy(parameters)
    comparator = create_population_comparator(parameters, periodic=periodic)
    shortcut = comparator.comparators[0] if method == "combined" else comparator
    assert np.all(shortcut.comparators[1].pbc == periodic)
    assert parameters == original
    with pytest.raises(ValueError, match="system.periodic"):
        create_population_comparator(dict(method="combined", comparators=[dict(method="energetic_oganov_fingerprint", pbc=True)]))


def test_selector_keeps_energy_criterion_in_prepared_comparison(tmp_path):
    comparator = dict(method="energetic_oganov_fingerprint", ene_diff_tol=1, pbc=False)
    selector = create_selector(dict(method="compare", number=2, rank_by="energy", comparator=comparator, directory=tmp_path, random_seed=0))
    selected = selector.select([candidate(-3), candidate(-2.8), candidate(-1)])
    assert [atoms.get_potential_energy() for atoms in selected] == [-3, -1]


@pytest.mark.parametrize("comparators", [[], {}, None])
def test_combined_requires_nonempty_criteria(comparators):
    with pytest.raises(ValueError, match="nonempty list"):
        create_comparator(dict(method="combined", comparators=comparators, random_seed=0))


def test_population_list_shorthand_and_signature_based_periodicity():
    configuration = [dict(method="energy"), dict(method="oganov_fingerprint")]
    original = copy.deepcopy(configuration)
    comparator = create_population_comparator(configuration, periodic=False)
    assert type(comparator) is CombinedComparator
    assert not comparator.comparators[1].pbc.any()
    assert configuration == original


def test_provisional_names_are_removed():
    for name in ("all", "energy_and_structure"):
        with pytest.raises(KeyError):
            create_comparator(dict(method=name))
