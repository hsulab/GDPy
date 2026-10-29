import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.population.comparators import create_population_comparator
from gdpx.exploration.genetic_algorithm.crossover import PeriodicCutAndSpliceCrossover
from gdpx.exploration.genetic_algorithm.mutation.rattle import RattleMutation
from gdpx.exploration.genetic_algorithm.mutation.mirror import MirrorMutation
from gdpx.structures.geometry.ga import closest_distances_generator


def candidate():
    atoms = Atoms('CuOCuO', positions=[[1, 1, 0], [2, 2, 4], [4, 4, 0], [4, 2, 4]],
                  tags=[0, 1, 0, 2], cell=[12, 12, 20], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-10)
    return atoms


@pytest.mark.parametrize('method,params', [('interatomic_distance', {}), ('atoms', {}),
                                         ('displacement', {'rmsd_max': 0.1}),
                                         ('oganov_fingerprint', {'rcut': 5})])
def test_comparison_excludes_interleaved_substrate(method, params):
    first = candidate()
    second = candidate()
    second.positions[[0, 2]] += [1, 0, 0]
    second.calc = SinglePointCalculator(second, energy=-10)
    comparator = create_population_comparator(dict(method=method, **params))
    assert comparator.looks_like(first, second)
    second.positions[3] += [1, 0, 0]
    second.calc = SinglePointCalculator(second, energy=-10)
    assert not comparator.looks_like(first, second)


def test_original_index_group_and_fingerprint_cache_are_consistent():
    first, second = candidate(), candidate()
    second.positions[3] += [1, 0, 0]
    second.calc = SinglePointCalculator(second, energy=-10)
    # Include one substrate index deliberately; tag 0 must still be excluded.
    selected = create_population_comparator(dict(method='interatomic_distance',
                                                  group_expression='`index 0 1`'))
    all_mobile = create_population_comparator(dict(method='interatomic_distance'))
    first.info['interatomic_distance'] = {'unrelated': 'cache'}
    for comparator, expected in [(selected, True), (all_mobile, False)]:
        assert comparator.looks_like(first, second) == expected
        comparator._precompute_fingerprint([first, second])
        assert comparator.looks_like(first, second) == expected
        comparator._delete_fingerprint([first, second])
        assert not comparator._prepared
    assert first.info['interatomic_distance'] == {'unrelated': 'cache'}


def test_energy_gate_uses_complete_candidate_energy():
    first, second = candidate(), candidate()
    second.calc = SinglePointCalculator(second, energy=-9)
    comparator = create_population_comparator(dict(method='interatomic_distance', dE=0.1))
    assert not comparator.looks_like(first, second)
    combined = create_population_comparator(dict(method='combined', comparators=[
        dict(method='energy', ene_diff_tol=0.1), dict(method='atoms')]))
    assert not combined.looks_like(first, second)


def test_empty_mobile_group_and_legacy_ntop_raise():
    with pytest.raises(ValueError, match='group_expression'):
        create_population_comparator(dict(method='interatomic_distance', n_top=2))
    comparator = create_population_comparator(dict(method='atoms', group_expression='`symbol Cu`'))
    with pytest.raises(ValueError, match='no mobile atoms'):
        comparator.looks_like(candidate(), candidate())


@pytest.mark.parametrize('use_tags', [False, True])
def test_crossover_uses_tags_not_array_tail(use_tags):
    first, second = candidate(), candidate()
    second.positions[[1, 3], 1] += 1
    slab = first[first.get_tags() == 0]
    crossover = PeriodicCutAndSpliceCrossover(slab=slab,
        blmin=closest_distances_generator(first.numbers, 0.3), use_tags=use_tags,
        rng=np.random.default_rng(42))
    original = first.positions.copy()
    child = crossover.cross(first, second)
    assert child is not None
    assert sorted(child.get_tags()) == [0, 0, 1, 2]
    np.testing.assert_allclose(child[child.get_tags() == 0].positions, slab.positions)
    np.testing.assert_array_equal(first.positions, original)


@pytest.mark.parametrize('mutation_type', [RattleMutation, MirrorMutation])
def test_mutations_select_nonzero_tags_without_engine_counts(mutation_type):
    parent = candidate()
    mutation = mutation_type(blmin=closest_distances_generator(parent.numbers, 0.3),
                             rng=np.random.default_rng(42))
    child = mutation.mutate(parent)
    assert child is not None
    assert sorted(child.get_tags()) == [0, 0, 1, 2]
    np.testing.assert_allclose(child[child.get_tags() == 0].positions,
                               parent[parent.get_tags() == 0].positions)
