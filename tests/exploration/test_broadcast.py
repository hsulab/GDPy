"""Explicit recipe sweeps expand before builders and search engines are created."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gdpx.exploration import REGISTER
from gdpx.exploration.factory import create_expedition
from gdpx.execution.factory import create_worker
from gdpx.utils.parser import parse_input_file

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples/global_optimisation'


@pytest.fixture
def capture(monkeypatch):
    for method in ('basin_hopping', 'monte_carlo', 'simulated_annealing'):
        monkeypatch.setitem(REGISTER._dict, method, lambda **kw: SimpleNamespace(params=kw))


def test_cartesian_order_paths_and_independent_overrides(capture):
    source = dict(method='basin_hopping', recipe=dict(random_seed=1, population={},
                  operators=[dict(temperature=500, particles=['Cu'])]),
                  broadcast={'random_seed': [7, 17], 'operators.0.temperature': [300, 600],
                             'operators.0.particles': [['Cu', 'Ni']]})
    original = copy.deepcopy(source)
    results = create_expedition(source)
    assert [(r.params['random_seed'], r.params['operators'][0]['temperature']) for r in results] == [
        (7, 300), (7, 600), (17, 300), (17, 600)]
    results[0].params['operators'][0]['particles'].append('Au')
    assert results[1].params['operators'][0]['particles'] == ['Cu', 'Ni']
    assert source == original


@pytest.mark.parametrize('method', ['monte_carlo', 'simulated_annealing'])
def test_optional_leaf_and_single_recipe_compatibility(capture, method):
    recipe = {'operators': [{'particles': ['Cu', 'Ni']}], 'random_seed': 1}
    assert not isinstance(create_expedition(dict(method=method, recipe=recipe)), list)
    result = create_expedition(dict(method=method, recipe=recipe, broadcast={'optional': [42]}))
    assert len(result) == 1
    assert result[0].params['optional'] == 42
    assert result[0].params['operators'][0]['particles'] == ['Cu', 'Ni']


@pytest.mark.parametrize('broadcast', [None, {}, [], {'random_seed': []}, {'random_seed': 7},
    {'unknown.seed': [7]}, {'operators.2.temperature': [300]}, {'operators.-1.temperature': [300]},
    {'operators.00.temperature': [300]}, {'random_seed.value': [7]}, {'': [7]}, {1: [7]},
    {'operators..temperature': [300]}, {'operators': [[]], 'operators.0.temperature': [300]},
    {'operators.0.temperature': [300], 'operators': [[]]}])
def test_invalid_broadcast_before_construction(monkeypatch, broadcast):
    monkeypatch.setitem(REGISTER._dict, 'monte_carlo', lambda **kw: pytest.fail('constructed invalid sweep'))
    with pytest.raises(ValueError):
        create_expedition(dict(method='monte_carlo', recipe={'random_seed': 1,
            'operators': [{'temperature': 500}]}, broadcast=broadcast))


def test_composition_replaces_entire_mapping(capture):
    source = dict(method='basin_hopping', recipe={'population': {'composition': {'Cu': 8, 'Au': 1}}},
                  broadcast={'population.composition': [{'Cu': 6, 'Ni': 2}, {'Cu': 4, 'Ni': 4}]})
    results = create_expedition(source)
    assert [r.params['population']['composition'] for r in results] == [{'Cu': 6, 'Ni': 2}, {'Cu': 4, 'Ni': 4}]
    results[0].params['population']['composition']['Cu'] = 99
    assert source['broadcast']['population.composition'][0]['Cu'] == 6
    assert results[1].params['population']['composition']['Cu'] == 4


def test_ga_implicit_broadcast_is_flattened_inside_explicit_sweep(monkeypatch):
    from gdpx.exploration.genetic_algorithm import engine
    monkeypatch.setattr(engine, 'GeneticAlgorithmEngine', lambda **kw: SimpleNamespace(params=kw))
    monkeypatch.setitem(REGISTER._dict, 'genetic_algorithm', engine.GeneticAlgorithmBroadcaster)
    source = parse_input_file(EXAMPLES / 'explorations/genetic_algorithm/cu7ni6.yaml')
    source['recipe']['objective'] = {'target': 'formation_energy', 'chemical_potentials': {'Cu': [-3, -2], 'Ni': -4}}
    source['broadcast'] = {'random_seed': [7, 17]}
    results = create_expedition(source)
    assert [(r.params['random_seed'], r.params['objective']['chemical_potentials']['Cu']) for r in results] == [
        (7, -3), (7, -2), (17, -3), (17, -2)]


def test_real_composition_example_serializes_resolved_recipes():
    source = parse_input_file(EXAMPLES / 'explorations/basin_hopping/cu_ni_compositions.yaml')
    runtime = parse_input_file(EXAMPLES / 'runtimes/emt.yaml')
    results = create_expedition(source)
    assert len(results) == 2
    assert results[0].rng is not results[1].rng
    for expedition, composition in zip(results, [{'Cu': 6, 'Ni': 2}, {'Cu': 4, 'Ni': 4}]):
        expedition.register_worker(create_worker(copy.deepcopy(runtime)))
        saved = expedition.as_dict()
        assert 'broadcast' not in saved
        assert saved['recipe']['population']['builders']['random']['composition'] == composition
        saved.pop('runtime')
        restored = create_expedition(saved)
        assert not isinstance(restored, list)
        assert restored.random_seed == 7
    assert results[0].worker is not results[1].worker
