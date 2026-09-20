"""Shared population semantics and ownership across BH and GA."""
import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.population.config import PopulationConfig
from gdpx.exploration.population.comparators import create_population_comparator
from gdpx.exploration.population.pool import CandidatePool, compute_population_fitness
from gdpx.exploration.population.random import RandomStreamRegistry
from gdpx.exploration.basin_hopping.population import HoppingPopulation
from gdpx.exploration.genetic_algorithm.population.manager import PopulationManager
from gdpx.exploration.genetic_algorithm.population.population import PopulationWithVariableComposition
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
    bh = HoppingPopulation(params, RandomStreamRegistry(7))
    ga_params = copy.deepcopy(params)
    ga_params["generation"]["completion"] = {"builder_proportions": [{"builder": "random", "proportion": 1.}]}
    ga = PopulationManager(ga_params)
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
        HoppingPopulation(params, RandomStreamRegistry(7))


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
    pool = CandidatePool(Database(frames), 5, comparator, use_extinct=True)
    assert [a.info["confid"] for a in pool.candidates] == [2, 3]
    assert pool.candidates[0] is frames[1]
    assert pool.candidates[0].info["looks_like"] == 1
    assert len(pool.select(9, np.random.default_rng(2))) == 9


def test_empty_singleton_equal_scores_and_bh_exact_chain_count():
    bh = HoppingPopulation(settings(retained=1, generation=5), RandomStreamRegistry(7))
    frame = candidate(1)
    selected = bh.get_current_generation(Database([frame]))
    assert len(selected) == 5 and all(a is frame for a in selected)
    assert bh.get_current_generation(Database([])) == []
    assert compute_population_fitness([]) == []
    assert compute_population_fitness([candidate(1)]) == [1.]
    assert compute_population_fitness([candidate(1), candidate(2)]) == [1., 1.]


def test_ga_retained_capacity_and_variable_selection():
    params = settings(initial=1, retained=4, generation=2)
    params["name"] = "variable"
    params["generation"]["completion"] = {"builder_proportions": [{"builder": "random", "proportion": 1.}]}
    manager = PopulationManager(params, rng=np.random.default_rng(9))
    frames = [candidate(1), candidate(2), candidate(3, symbol="Ag"), candidate(4, symbol="Ag")]
    manager.update_population(Database(frames), create_population_comparator({"method": "atoms"}))
    assert len(manager.population.pop) == 4
    assert isinstance(manager.population, PopulationWithVariableComposition)
    for _ in range(20):
        first, second = manager.population.get_two_candidates()
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


@pytest.mark.parametrize("example", ["cu8_bh_emt.yaml", "cu13_emt.yaml"])
def test_example_population_round_trip_and_config_immutability(example):
    from pathlib import Path
    import yaml
    from gdpx.exploration.factory import create_expedition
    from gdpx.execution.factory import create_worker

    source = Path(__file__).resolve().parents[2] / "examples/global_optimisation" / example
    config = yaml.safe_load(source.read_text())
    runtime = config.pop("runtime")
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
    population = getattr(rebuilt, "pop_manager", None) or rebuilt.population
    assert (population.init_size, population.retained_size, population.gen_size) == (4, 2, 2)
