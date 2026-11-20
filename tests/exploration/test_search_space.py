"""Explicit atom roles shared by population algorithms and operators."""
import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.population.search_space import SearchSpace, membership, transfer_membership
from gdpx.exploration.population.comparators import create_population_comparator
from gdpx.exploration.population.operators import bind_genetic_group
from gdpx.exploration.genetic_algorithm.mutation.rattle import RattleMutation
from gdpx.structures.groups import evaluate_group_expression


def atoms():
    return Atoms('Cu3', positions=[[0, 0, 0], [4, 0, 0], [8, 0, 0]],
                 cell=[20, 20, 20], tags=[0, 0, 1])


def runtime(constraint=None):
    setup = {} if constraint is None else {'constraint': constraint}
    return {'potential': {'provider': 'emt'}, 'executor': {
        'provider': 'ase', 'method': 'min', 'parameters': {
            'setup': setup, 'stop': {'steps': 20, 'fmax': .1}}}}


def test_compatibility_defaults_and_no_tags():
    policy = SearchSpace()
    a = atoms()
    assert policy.select(a) == [2]
    assert policy.select(a, 'buffer or frozen') == []
    b = Atoms('Cu2')
    assert policy.select(b) == [0, 1]
    assert not b.has('tags')


def test_named_groups_are_fixed_after_initial_selection():
    policy = SearchSpace({'active': 'not tag 0', 'buffer': 'index 1', 'frozen': 'index 0'})
    rt = policy.prepare_runtime(runtime())
    assert rt['executor']['parameters']['setup']['constraint'] == 'index 0'
    policy.bind_runtime(rt)
    a = atoms()
    assert policy.select(a, 'active or buffer') == [1, 2]
    a.positions[1, 2] = 100
    assert evaluate_group_expression(a, 'buffer') == [1]
    saved = copy.deepcopy(a.info)
    b = Atoms(a.symbols, positions=a.positions, tags=a.get_tags(), info=saved)
    assert policy.select(b, 'active or buffer') == [1, 2]


def test_position_membership_survives_backend_and_database(tmp_path):
    from gdpx.exploration.persist.database import GlobalOptimisationDatabase
    policy = SearchSpace({'active': 'ztop 1', 'buffer': 'index 1'})
    source = atoms()
    source.positions[:, 2] = [0, 1, 2]
    policy.resolve(source)
    output = Atoms(source.symbols, positions=source.positions, tags=source.get_tags())
    output.positions[2, 2] = -10
    transfer_membership(source, output)
    output.info['key_value_pairs'] = {'raw_score': 1, 'generation': 0}
    output.calc = SinglePointCalculator(output, energy=-1)
    database = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    database.add_evaluated_candidate(output, 'test')
    restored, = database.get_all_relaxed_candidates()
    assert policy.select(restored) == [2]


def test_frozen_constraints_inferred_and_compared_by_indices():
    policy = SearchSpace({'frozen': 'tag 0'})
    policy.bind_runtime(runtime('index 0 1'))
    assert policy.select(atoms(), 'frozen') == [0, 1]
    wrong = SearchSpace({'frozen': 'index 0'})
    wrong.bind_runtime(runtime('index 1'))
    with pytest.raises(ValueError, match='inconsistent'):
        wrong.resolve(atoms())
    inferred = SearchSpace()
    inferred.bind_runtime(runtime('index 2'))
    assert inferred.select(atoms()) == [2]
    assert inferred.select(atoms(), movable=True) == []


@pytest.mark.parametrize('definition', [
    {'unexpected': 'not tag 0'}, {'active': []}, {'buffer': ''},
])
def test_invalid_configuration(definition):
    with pytest.raises(ValueError):
        SearchSpace(definition)


@pytest.mark.parametrize('definition', [
    {'active': 'index 1', 'buffer': 'index 1'},
    {'active': 'index 1', 'frozen': 'index 1'},
])
def test_explicit_groups_do_not_overlap(definition):
    with pytest.raises(ValueError, match='overlap'):
        SearchSpace(definition).resolve(atoms())


def test_comparator_defaults_active_and_can_include_zero_tag_buffer():
    policy = SearchSpace({'buffer': 'index 1'})
    first, second = atoms(), atoms()
    second.positions[1, 0] += 1
    default = create_population_comparator({'method': 'atoms'}, search_space=policy)
    context = create_population_comparator({'method': 'atoms', 'group_expression': 'active or buffer'}, search_space=policy)
    assert default.looks_like(first, second)
    assert not context.looks_like(first, second)


@pytest.mark.parametrize('group,changed', [(None, [2]), ('active or buffer', [1, 2])])
def test_ga_mutation_respects_group_and_preserves_tags(group, changed):
    policy = SearchSpace({'buffer': 'index 1', 'frozen': 'index 0'})
    parent = atoms()
    parent.info['confid'] = 1
    before = parent.copy()
    op = RattleMutation({(29, 29): .1}, rattle_strength=.1, rattle_prop=1, rng=np.random.default_rng(1))
    bind_genetic_group(op, policy, group)
    child, _ = op.get_new_individual([parent])
    assert child is not None
    assert np.flatnonzero(np.any(child.positions != before.positions, axis=1)).tolist() == changed
    assert child.get_tags().tolist() == [0, 0, 1]
    assert policy.select(child, 'buffer') == [1]
    np.testing.assert_array_equal(parent.positions, before.positions)
    np.testing.assert_array_equal(parent.get_tags(), before.get_tags())


@pytest.mark.parametrize('group,changed', [(None, [2]), ('active or buffer', [1, 2])])
def test_bh_rattle_respects_group_and_rollback(group, changed):
    from gdpx.sampling.moves import RattleOperator
    policy = SearchSpace({'buffer': 'index 1', 'frozen': 'index 0'})
    parent = atoms()
    before = parent.copy()
    op = RattleOperator(particles=['Cu'], rattle_strength=.1, rattle_prop=1,
                        group=group, skip_distance_check=True, allow_isolated=True)
    op.search_space = policy
    proposal = op.propose(parent, np.random.default_rng(2))
    assert proposal.valid
    assert np.flatnonzero(np.any(parent.positions != before.positions, axis=1)).tolist() == changed
    assert parent.get_tags().tolist() == [0, 0, 1]
    proposal.rollback()
    np.testing.assert_array_equal(parent.positions, before.positions)
    np.testing.assert_array_equal(parent.get_tags(), before.get_tags())


@pytest.mark.parametrize('method', ['genetic_algorithm', 'basin_hopping'])
def test_population_search_preserves_roles_through_generation_and_restart(tmp_path, method):
    from pathlib import Path
    import yaml
    from gdpx.exploration.factory import create_exploration
    from gdpx.execution.factory import create_worker
    root = Path(__file__).resolve().parents[2] / 'examples/global_optimisation'
    config = yaml.safe_load((root / method / 'cu8.yaml').read_text())
    config['system']['initial'] = {'total_size': 2, 'builder_allocations': [{'builder': 'init', 'size': 2}]}
    config['system']['generation']['total_size'] = 1
    config['system']['search_space'] = {'active': 'not (tag 0 or tag 1)', 'frozen': 'tag 1'}
    config['strategy']['convergence'] = {'generation': 1}
    config['strategy']['use_archive'] = False
    if method == 'genetic_algorithm':
        config['strategy']['reproduction']['size'] = 1
    else:
        config['strategy']['steps_per_chain'] = 1
    rt = runtime()
    rt['executor']['parameters']['stop']['steps'] = 100

    def construct():
        obj = create_exploration(copy.deepcopy(config))
        obj = obj[0] if isinstance(obj, list) else obj
        obj.directory = tmp_path / method
        obj.register_worker(create_worker(copy.deepcopy(rt)))
        return obj

    obj = construct()
    obj.run()
    resumed = construct()
    resumed.run()
    assert resumed.read_convergence()
    from gdpx.exploration.persist.database import GlobalOptimisationDatabase
    database = GlobalOptimisationDatabase(resumed.directory / 'candidates.db')
    resumed.population.refresh(database)
    assert resumed.population.candidates
    for candidate in resumed.population.candidates:
        groups = membership(candidate)
        assert groups is not None
        assert groups['active'].sum() == 7
        assert groups['frozen'].sum() == 1


def test_bh_exchange_assigns_new_atoms_active_and_rolls_back_roles():
    from gdpx.sampling.moves import ExchangeOperator
    policy = SearchSpace({'buffer': 'index 1', 'frozen': 'index 0'})
    parent = atoms()
    policy.resolve(parent)
    original = {key: value.copy() for key, value in parent.arrays.items()}
    op = ExchangeOperator(particles=['Cu'], chempots=[0.], skip_distance_check=True,
                          allow_isolated=True)
    op.search_space = policy
    op.bond_distance_dict = {(29, 29): 2.0}
    op.custom_pair_distance_dict = None
    op.blmin = {(29, 29): 1.6}
    inserted = False
    for seed in range(10):
        proposal = op.propose(parent, np.random.default_rng(seed))
        if proposal.valid:
            if len(parent) > 3:
                inserted = True
                assert parent.arrays['gdpx_active'][-1]
                assert not parent.arrays['gdpx_frozen'][-1]
            proposal.rollback()
        for key, expected in original.items():
            np.testing.assert_array_equal(parent.arrays[key], expected)
    assert inserted


@pytest.mark.parametrize('group', [None, 'active or buffer'])
def test_ga_crossover_preserves_excluded_atoms_and_fragment_tags(group):
    from gdpx.exploration.genetic_algorithm.crossover import PeriodicCutAndSpliceCrossover
    from gdpx.structures.geometry.ga import closest_distances_generator
    first = Atoms('CuOCuO', positions=[[1, 1, 0], [2, 2, 4], [4, 4, 0], [4, 2, 4]],
                  tags=[0, 1, 0, 2], cell=[12, 12, 20], pbc=True)
    second = first.copy()
    second.positions[[1, 3], 1] += 1
    first.info['confid'], second.info['confid'] = 1, 2
    policy = SearchSpace({'buffer': 'index 2', 'frozen': 'index 0'})
    operator = PeriodicCutAndSpliceCrossover(
        slab=first[first.get_tags() == 0], use_tags=True,
        blmin=closest_distances_generator(first.numbers, .3), rng=np.random.default_rng(42))
    bind_genetic_group(operator, policy, group)
    child, _ = operator.get_new_individual([first, second])
    assert child is not None
    assert child.get_tags().tolist() == first.get_tags().tolist()
    np.testing.assert_array_equal(child.positions[0], first.positions[0])
    if group is None:
        np.testing.assert_array_equal(child.positions[2], first.positions[2])
    assert policy.select(child, 'buffer') == [2]
