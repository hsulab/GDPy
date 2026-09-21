"""Shared population semantics and ownership across BH and GA."""
import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.population.config import PopulationConfig
from gdpx.exploration.population.comparators import create_population_comparator
from gdpx.exploration.population import Population
from gdpx.exploration.population.population import compute_population_fitness
from gdpx.exploration.population.random import RandomStreamRegistry
from gdpx.exploration.basin_hopping.selection import HoppingStartSelector
from gdpx.exploration.genetic_algorithm.generation import GeneticGenerationManager
from gdpx.exploration.genetic_algorithm.selection import GeneticParentSelector
from gdpx.exploration.genetic_algorithm.engine import GeneticAlgorithmEngine
from gdpx.exploration.persist.database import GlobalOptimisationDatabase


def settings(initial=2, retained=3, generation=4):
    return {
        "retained_size": retained, "periodic": False, "preserve_fragments": False,
        "builders": {"random": {"method": "random_structure_improved", "composition": {"Cu": 2},
                                "box": [10., 10., 10.]}},
        "initial": {"total_size": initial, "builder_allocations": [{"builder": "random", "size": initial}]},
        "generation": {"total_size": generation}, "comparator": {"method": "atoms"},
    }


def candidate(index, score=1., position=None, symbol="Cu", extinct=False):
    atoms = Atoms(symbol, positions=[[index if position is None else position, 0, 0]], tags=[1])
    atoms.info = {"confid": index, "key_value_pairs": {"raw_score": score, "extinct": int(extinct)}}
    atoms.calc = SinglePointCalculator(atoms, energy=-score)
    return atoms


class Database:
    def __init__(self, frames):
        self.frames = frames

    def get_all_relaxed_candidates(self, use_extinct=False):
        return [a for a in self.frames if not use_extinct or not a.info["key_value_pairs"]["extinct"]]

    def get_participation_in_pairing(self):
        return {}, []


def test_both_methods_accept_independent_sizes_without_mutating_config():
    params = settings()
    original = copy.deepcopy(params)
    bh = PopulationConfig(params)
    ga_params = copy.deepcopy(params)
    ga_params["generation"]["completion"] = {"builder_proportions": [{"builder": "random", "proportion": 1.}]}
    ga = PopulationConfig(ga_params)
    for manager in [bh, ga]:
        assert (manager.init_size, manager.retained_size, manager.gen_size) == (2, 3, 4)
    assert params == original
    del params["retained_size"]
    assert PopulationConfig(params).retained_size == 4


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_retained_size_requires_positive_integer(value):
    with pytest.raises(ValueError, match="retained_size"):
        PopulationConfig(settings(retained=value))


@pytest.mark.parametrize("old,new", [("initial_size", "initial.total_size"),
                                     ("generation_size", "generation.total_size"),
                                     ("population_size", "retained_size"),
                                     ("random_offspring_generator", "builders")])
def test_bh_migration_names_replacement(old, new):
    params = settings()
    params[old] = 1
    with pytest.raises(ValueError, match=new):
        PopulationConfig(params)


def test_comparator_location_migration():
    for operators in [{"comparator": {}}, {"mobile": {"comparator": {}}}]:
        with pytest.raises(ValueError, match="population.comparator"):
            GeneticAlgorithmEngine(population=settings(), operators=operators, convergence={})


def test_pool_ranks_deduplicates_counts_history_and_borrows(monkeypatch):
    frames = [candidate(1, 1, 0), candidate(2, 3, 0), candidate(3, 2, 2), candidate(4, 4, 4, extinct=True)]
    def no_copy(*args, **kwargs):
        raise AssertionError("selection must not copy Atoms")
    monkeypatch.setattr(Atoms, "copy", no_copy)
    comparator = create_population_comparator({"method": "atoms"}, False)
    pool = Population(5, comparator, use_extinct=True)
    pool.refresh(Database(frames))
    assert [a.info["confid"] for a in pool.candidates] == [2, 3]
    assert pool.candidates[0] is frames[1]
    assert pool.similarity_counts[2] == 1
    assert len(HoppingStartSelector(np.random.default_rng(2)).select(pool, 9)) == 9


def test_empty_singleton_equal_scores_and_bh_exact_chain_count():
    bh = Population(1, create_population_comparator({"method": "atoms"}))
    selector = HoppingStartSelector(RandomStreamRegistry(7).get("population"))
    frame = candidate(1)
    bh.refresh(Database([frame]))
    selected = selector.select(bh, 5)
    assert len(selected) == 5 and all(a is frame for a in selected)
    bh.refresh(Database([]))
    assert selector.select(bh, 5) == []
    assert compute_population_fitness([]).size == 0
    assert compute_population_fitness([candidate(1)]) == [1.]
    np.testing.assert_array_equal(compute_population_fitness([candidate(1), candidate(2)]), [1., 1.])


def test_ga_retained_capacity_and_variable_selection():
    params = settings(initial=1, retained=4, generation=2)
    params["name"] = "variable"
    params["generation"]["completion"] = {"builder_proportions": [{"builder": "random", "proportion": 1.}]}
    config = PopulationConfig(params)
    population = Population(config.retained_size, create_population_comparator({"method": "atoms"}))
    rng = np.random.default_rng(9)
    selector = GeneticParentSelector(rng, "variable")
    manager = GeneticGenerationManager(params, config, population, selector, rng)
    frames = [candidate(1), candidate(2), candidate(3, symbol="Ag"), candidate(4, symbol="Ag")]
    manager.update_population(Database(frames))
    assert len(population.candidates) == 4
    assert type(manager.population) is Population
    for _ in range(20):
        first, second = selector.select_pair(population)
        assert first is not second
        assert first.symbols == second.symbols
        assert any(first is a for a in frames)



class Builder:
    def __init__(self, frame=None):
        self.frame = frame

    def run(self, size):
        return [] if self.frame is None else [self.frame for _ in range(size)]


def test_initial_allocations_preserve_arrays_and_own_results():
    config = settings(initial=2)
    config["initial"]["builder_allocations"] = [{"builder": "a", "size": 1}, {"builder": "b", "size": 1}]
    manager = PopulationConfig(config)
    frame = candidate(1)
    frame.new_array("custom", np.array([3.]))
    result = manager._prepare_initial_population({"a": Builder(frame), "b": Builder(frame)})
    assert [a.info["data"]["builder"] for a in result] == ["a", "b"]
    result[0].positions[0, 0] = 100
    result[0].arrays["custom"][0] = 5
    assert frame.positions[0, 0] == result[1].positions[0, 0] == 1
    assert frame.arrays["custom"][0] == result[1].arrays["custom"][0] == 3
    with pytest.raises(RuntimeError, match="generated 0 of 1"):
        manager._generate_from_builder("empty", Builder(), 1, 2)


@pytest.mark.parametrize("legacy", [False, True])
def test_generation_accounting_uses_generation_size_not_retained(tmp_path, legacy):
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    data = {"initial_population_size": 1, "retained_size": 8, "num_atoms_substrate": 0,
            "population_size" if legacy else "generation_size": 2}
    db.init_task(Atoms(), data)
    for index, gen in enumerate([0, 1, 1]):
        db.connection.write(candidate(index), relaxed=1, generation=gen)
    assert db.get_generation_number() == 2


@pytest.mark.parametrize("example, runtime_name", [
    ("explorations/basin_hopping/cu8.yaml", "emt_min_300.yaml"),
    ("explorations/genetic_algorithm/cu13.yaml", "emt_min_100.yaml"),
])
def test_example_population_round_trip_and_config_immutability(example, runtime_name):
    from pathlib import Path
    import yaml
    from gdpx.exploration.factory import create_expedition
    from gdpx.execution.factory import create_worker

    source = Path(__file__).resolve().parents[2] / "examples/global_optimisation" / example
    config = yaml.safe_load(source.read_text())
    runtime = yaml.safe_load((source.parents[2] / "runtimes" / runtime_name).read_text())
    original = copy.deepcopy(config)
    engine = create_expedition(copy.deepcopy(config))
    if isinstance(engine, list):
        engine = engine[0]
    engine.register_worker(create_worker(runtime))
    saved = engine.as_dict()
    assert config == original
    assert saved["recipe"]["population"]["retained_size"] == 2
    assert saved["recipe"]["population"]["comparator"] == {"method": "interatomic_distance"}
    saved.pop("runtime")
    rebuilt = create_expedition(saved)
    if isinstance(rebuilt, list):
        rebuilt = rebuilt[0]
    assert type(rebuilt.population) is Population
    population = rebuilt.population_config
    assert (population.init_size, population.retained_size, population.gen_size) == (4, 2, 2)


def selection_fixture_database():
    """Inputs used to record selection before separating populations and selectors."""
    frames = []
    entries = [("Cu", 2., 4.), ("Cu", 2.2, 3.), ("Cu", 2.4, 2.),
               ("Ag", 2., 3.5), ("Ag", 2.2, 1.), ("Cu", 2., 0.)]
    for index, (symbol, distance, score) in enumerate(entries, 1):
        atoms = Atoms(symbol + "2", positions=[[0, 0, 0], [distance, 0, 0]], tags=[1, 2])
        atoms.info = {"confid": index, "key_value_pairs": {"raw_score": score}, "data": {"source": "fixture"}}
        frames.append(atoms)
    database = Database(frames)
    database.get_participation_in_pairing = lambda: ({1: 8, 2: 2, 4: 4}, [(1, 2), (1, 4)])
    return database


@pytest.mark.parametrize("policy", ["constant", "variable", "bh"])
@pytest.mark.parametrize("with_history", [True, False])
def test_selection_and_rng_match_before_refactor(policy, with_history):
    import json
    from pathlib import Path

    baseline = json.loads((Path(__file__).parent / "fixtures/population_selection.json").read_text())
    database = selection_fixture_database()
    original_info = [copy.deepcopy(a.info) for a in database.frames]
    streams = RandomStreamRegistry(37)
    population = Population(5, create_population_comparator({"method": "atoms"}))
    population.refresh(database)
    draws = []
    if policy == "bh":
        selector = HoppingStartSelector(streams.get("population"))
        for _ in range(8):
            draws.append([a.info["confid"] for a in selector.select(population, 7, with_history)])
    else:
        selector = GeneticParentSelector(streams.get("population"), policy)
        selector.refresh(population, database)
        for _ in range(8):
            pair = selector.select_pair(population, with_history)
            one = selector.select_one(population, with_history)
            draws.append({"pair": [a.info["confid"] for a in pair], "one": one.info["confid"]})
    assert {"draws": draws, "state": streams.snapshot()} == baseline[f"{policy}_{with_history}"]
    assert [a.info for a in database.frames] == original_info


@pytest.mark.parametrize("policy", ["constant", "variable"])
def test_ga_selection_empty_singleton_and_missing_compatible_pair(policy):
    rng = np.random.default_rng(5)
    selector = GeneticParentSelector(rng, policy)
    population = Population(4, create_population_comparator({"method": "atoms"}))
    database = Database([])
    population.refresh(database)
    selector.refresh(population, database)
    state = copy.deepcopy(rng.bit_generator.state)
    assert selector.select_pair(population) is None
    assert selector.select_one(population) is None
    assert rng.bit_generator.state == state

    database.frames = [candidate(1)]
    population.refresh(database)
    selector.refresh(population, database)
    assert selector.select_pair(population) is None
    assert selector.select_one(population) is database.frames[0]
    database.frames.append(candidate(2, symbol="Ag"))
    population.refresh(database)
    selector.refresh(population, database)
    if policy == "variable":
        assert selector.select_pair(population) is None
    else:
        assert len(selector.select_pair(population)) == 2


def test_refresh_rebuilds_membership_statistics_and_ga_history():
    frames = [candidate(1, 1., 0.), candidate(2, 2., 0.), candidate(3, 3., 3.)]
    database = Database(frames)
    population = Population(2, create_population_comparator({"method": "atoms"}), use_extinct=True)
    selector = GeneticParentSelector(np.random.default_rng(1), "variable")
    population.refresh(database)
    selector.refresh(population, database)
    assert [a.info["confid"] for a in population.candidates] == [3, 2]
    assert dict(population.similarity_counts) == {3: 0, 2: 1}
    frames[2].info["key_value_pairs"]["extinct"] = 1
    frames.append(candidate(4, 4., 4., symbol="Ag"))
    database.get_participation_in_pairing = lambda: ({4: 3}, [])
    population.refresh(database)
    selector.refresh(population, database)
    assert [a.info["confid"] for a in population.candidates] == [4, 2]
    assert dict(population.similarity_counts) == {4: 0, 2: 1}
    assert selector.participation == {4: 3}
    assert [group[0].info["confid"] for group in selector.groups] == [4, 2]
    with pytest.raises(AttributeError):
        population.candidates = ()


@pytest.mark.parametrize("fail", [False, True])
def test_refresh_preserves_metadata_and_caches_even_on_failure(monkeypatch, fail):
    class CachingComparator:
        def _precompute_fingerprint(self, frames):
            for frame in frames:
                frame.info["fingerprint"] = np.ones(3)
        def _delete_fingerprint(self, frames):
            for frame in frames:
                frame.info.pop("fingerprint", None)
        def looks_like(self, first, second):
            if fail:
                raise RuntimeError("comparison failed")
            return False

    frames = [candidate(1), candidate(2)]
    frames[0].info["fingerprint"] = np.arange(3)
    frames[0].info["data"] = {"original": [1, 2]}
    originals = [(a.info, dict(a.info), a.positions, a.calc) for a in frames]
    population = Population(2, CachingComparator())
    monkeypatch.setattr(Atoms, "copy", lambda *args, **kwargs: pytest.fail("unexpected Atoms copy"))
    if fail:
        with pytest.raises(RuntimeError, match="comparison failed"):
            population.refresh(Database(frames))
        assert population.candidates == ()
    else:
        population.refresh(Database(frames))
        selector = GeneticParentSelector(np.random.default_rng(4))
        selector.refresh(population, Database(frames))
        selector.select_pair(population)
        HoppingStartSelector(np.random.default_rng(4)).select(population, 3)
    for atoms, (info, entries, positions, calc) in zip(frames, originals):
        assert atoms.info is info and atoms.info.keys() == entries.keys()
        assert all(atoms.info[key] is value for key, value in entries.items())
        assert atoms.positions is positions and atoms.calc is calc


def test_ga_incompatible_pairs_fall_back_to_independent_single_parent():
    from types import SimpleNamespace

    params = settings(initial=2, retained=2, generation=1)
    params["name"] = "variable"
    params["generation"].update({
        "reproduction": {"size": 1},
        "completion": {"builder_proportions": [{"builder": "random", "proportion": 1.}]},
    })
    config = PopulationConfig(params)
    frames = [candidate(1), candidate(2, symbol="Ag")]
    for frame in frames:
        frame.info["nested"] = {"unchanged": True}
    snapshots = [copy.deepcopy(a.info) for a in frames]
    positions = [a.positions.copy() for a in frames]
    population = Population(2, create_population_comparator({"method": "atoms"}))
    rng = np.random.default_rng(8)
    selector = GeneticParentSelector(rng, "variable")
    manager = GeneticGenerationManager(params, config, population, selector, rng)
    manager.update_population(Database(frames))
    writes = []
    database = SimpleNamespace(
        add_unrelaxed_candidate=lambda atoms, **kwargs: writes.append(atoms),
        add_unrelaxed_step=lambda atoms, description: writes.append(atoms),
    )

    class Mutation:
        oplist = [object()]

        def get_new_individual(self, parents):
            child = parents[0]
            assert all(child is not frame for frame in frames)
            child.positions += 0.1
            child.info["nested"]["unchanged"] = False
            return child, "mutation: fixture"

    pairing = SimpleNamespace(allow_variable_composition=False)
    child = manager._reproduce(database, 1, population,
                               {"mobile": {"pairing": pairing, "mutations": Mutation()}}, 0)
    assert child is not None and len(writes) == 2
    assert child.info["data"]["parents"][0] == child.info["data"]["parents"][1]
    for frame, info, coords in zip(frames, snapshots, positions):
        assert frame.info == info
        np.testing.assert_array_equal(frame.positions, coords)


@pytest.mark.parametrize('available,required', [(4, 2), (2, 2), (1, 3), (0, 2), (2, 0)])
def test_bh_replacement_defaults_and_borrowing(available, required, monkeypatch):
    from types import SimpleNamespace
    frames = tuple(candidate(i) for i in range(available))
    population = SimpleNamespace(candidates=frames, similarity_counts={})
    def no_copy(*args, **kwargs):
        pytest.fail('selection copied Atoms')
    monkeypatch.setattr(Atoms, 'copy', no_copy)
    rng = np.random.default_rng(17)
    selector = HoppingStartSelector(rng)
    result = selector.select(population, required)
    assert len(result) == (required if available else 0)
    assert all(any(frame is original for original in frames) for frame in result)
    if available >= required:
        assert len({id(frame) for frame in result}) == required
    expected = HoppingStartSelector(np.random.default_rng(17)).select(population, required)
    assert [id(a) for a in result] == [id(a) for a in expected]


def test_bh_explicit_replacement_keeps_weighted_legacy_draws():
    from types import SimpleNamespace
    frames = tuple(candidate(i, score=i) for i in range(4))
    population = SimpleNamespace(candidates=frames, similarity_counts={})
    weights = compute_population_fitness(frames, {})
    expected_rng = np.random.default_rng(7)
    actual_rng = np.random.default_rng(7)
    expected = expected_rng.choice(4, size=4, replace=True, p=weights / weights.sum())
    result = HoppingStartSelector(actual_rng, replace=True).select(population, 4)
    assert [a.info['confid'] for a in result] == list(expected)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    assert len(set(expected)) < 4
