from types import SimpleNamespace

import pytest
from ase import Atoms

from gdpx.exploration import REGISTER
from gdpx.exploration.factory import create_expedition
from gdpx.exploration.genetic_algorithm.engine import (
    GeneticAlgorithmBroadcaster,
    GeneticAlgorithmEngine,
)
from gdpx.exploration.genetic_algorithm.population.manager import PopulationManager
from gdpx.exploration.persist.database import GlobalOptimisationDatabase
from gdpx.exploration.monte_carlo.concurrent_hopping import ConcurrentHopping
from gdpx.exploration.monte_carlo.monte_carlo import MonteCarlo
from gdpx.exploration.monte_carlo.utils import parse_operators
from gdpx.exploration.simulated_annealing.simulated_annealing import SimulatedAnnealing


class Serializable:
    def __init__(self, value):
        self.value = value

    def as_dict(self):
        return self.value


@pytest.mark.parametrize(
    "method",
    [
        "genetic_algorithm",
        "monte_carlo",
        "basin_hopping",
        "concurrent_hopping",
        "simulated_annealing",
    ],
)
def test_factory_unpacks_recipe_and_keeps_seed(monkeypatch, method):
    captured = {}
    expedition = object()

    def create(**kwargs):
        captured.update(kwargs)
        return expedition

    monkeypatch.setitem(REGISTER._dict, method, create)

    result = create_expedition(
        {
            "method": method,
            "recipe": {
                "random_seed": 17,
                "operators": [],
                "convergence": {"steps": 2},
            },
        }
    )

    assert result is expedition
    assert captured == {
        "random_seed": 17,
        "operators": [],
        "convergence": {"steps": 2},
    }


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"method": "genetic_algorithm", "builder": {}, "params": {}},
            "requires a 'recipe' mapping",
        ),
        (
            {
                "method": "monte_carlo",
                "random_seed": 7,
                "recipe": {"operators": [], "convergence": {}},
            },
            "move these top-level keys into 'recipe': random_seed",
        ),
        ({"method": "simulated_annealing", "recipe": []}, "must be a mapping"),
    ],
)
def test_factory_rejects_legacy_global_optimisation_shapes(config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        create_expedition(config)


def test_ga_broadcaster_uses_named_recipe_fields():
    broadcaster = GeneticAlgorithmBroadcaster(
        population={
            "builders": {"primary": {"method": "unused"}},
            "reference_builder": "primary",
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "primary", "size": 1}]},
            "generation": {
                "total_size": 1,
                "reproduction": {"size": 1},
                "mutation": {"size": 0},
                "completion": {"builder_proportions": [{"builder": "primary", "proportion": 1.0}]},
            },
        },
        convergence={"generation": 1},
        property={
            "target": "formation_energy",
            "chempot": {"Cu": [-3.0, -2.0]},
        },
        random_seed=23,
    )

    assert len(broadcaster.input_params_list) == 2
    assert {item["property"]["chempot"]["Cu"] for item in broadcaster.input_params_list} == {-3.0, -2.0}
    assert all(item["random_seed"] == 23 for item in broadcaster.input_params_list)
    assert all("params" not in item for item in broadcaster.input_params_list)


def test_ga_population_uses_expanded_keys():
    population = PopulationManager(
        {
            "initial": {
                "total_size": 4,
                "builder_allocations": [
                    {"builder": "imported", "size": 1},
                    {"builder": "compact", "size": 3},
                ],
            },
            "generation": {
                "total_size": 4,
                "reproduction": {
                    "size": 2,
                    "maximum_attempts": 24,
                    "mutation_probability": 0.25,
                    "custom_mutation_probability": 0.75,
                },
                "mutation": {"size": 1, "maximum_attempts": 12},
                "completion": {
                    "builder_proportions": [
                        {"builder": "compact", "proportion": 0.75},
                        {"builder": "alternative", "proportion": 0.25},
                    ]
                },
            },
            "substrate": {"distance_tolerance": 0.1},
        }
    )

    assert population.init_size == 4
    assert population.gen_rep_size == 2
    assert population.gen_mut_size == 1
    assert population.gen_mut_max_try == 12
    assert population.gen_rep_max_try == 24
    assert population.pmut == 0.25
    assert population.pmut_custom == 0.75
    assert population.substrate_dtol == 0.1
    assert population.allocate_completion_sizes(6) == [
        {"builder": "compact", "size": 5, "maximum_attempts": 50},
        {"builder": "alternative", "size": 1, "maximum_attempts": 10},
    ]


@pytest.mark.parametrize(
    "population",
    [
        {"init": {"size": 4}},
        {"generation": {"size": 4, "reprod": 4}},
        {
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "a", "size": 1}]},
            "generation": {
                "total_size": 1,
                "completion": {"builder_proportions": [{"builder": "a", "proportion": 1.0}]},
            },
            "substrate": {"dtol": 0.1},
        },
    ],
)
def test_ga_population_rejects_abbreviated_keys(population):
    with pytest.raises(ValueError, match="Legacy GA"):
        PopulationManager(population)


def test_monte_carlo_operator_probability_is_expanded():
    operators, weights = parse_operators(
        [
            {"method": "move", "particles": ["H"], "probability": 1.0},
            {"method": "move", "particles": ["H"], "probability": 3.0},
        ]
    )

    assert weights == [0.25, 0.75]
    assert operators[0].as_dict()["probability"] == 1.0
    with pytest.raises(ValueError, match="probability"):
        parse_operators([{"method": "move", "particles": ["H"], "prob": 1.0}])


def test_ga_serialization_uses_recipe_and_runtime():
    engine = object.__new__(GeneticAlgorithmEngine)
    engine.random_seed = 7
    engine.builders = {
        "compact": Serializable({"method": "random_structure_improved"}),
        "imported": Serializable({"method": "direct"}),
    }
    engine.reference_builder_name = "compact"
    engine.worker = Serializable({"schema_version": 2})
    engine.ga_dict = {
        "database": "search.db",
        "population": {
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "compact", "size": 1}]},
            "generation": {"total_size": 1},
        },
        "operators": {},
        "property": {"target": "energy"},
        "convergence": {"generation": 1},
        "use_archive": True,
    }

    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 7
    assert config["recipe"]["population"]["builders"] == {
        "compact": {"method": "random_structure_improved"},
        "imported": {"method": "direct"},
    }
    assert config["recipe"]["population"]["reference_builder"] == "compact"
    assert "params" not in config
    assert "worker" not in config


class FixedBuilder:
    def __init__(self, symbol):
        self.symbol = symbol

    def run(self, size):
        return [Atoms(self.symbol) for _ in range(size)]


def test_initial_population_uses_ordered_builder_allocations():
    population = PopulationManager(
        {
            "initial": {
                "total_size": 3,
                "builder_allocations": [
                    {"builder": "first", "size": 2},
                    {"builder": "second", "size": 1},
                ],
            },
            "generation": {
                "total_size": 3,
                "completion": {"builder_proportions": [{"builder": "first", "proportion": 1.0}]},
            },
        }
    )
    frames = population._prepare_initial_population(
        {"first": FixedBuilder("H"), "second": FixedBuilder("He")}
    )

    assert [atoms.get_chemical_formula() for atoms in frames] == ["H", "H", "He"]
    assert [atoms.info["data"]["builder"] for atoms in frames] == ["first", "first", "second"]


def test_generation_plan_round_trip(tmp_path):
    database = GlobalOptimisationDatabase(tmp_path / "ga.db")
    database.init_task(
        Atoms("H"),
        data={"population_size": 2, "initial_population_size": 2, "num_atoms_substrate": 1},
    )
    plan = {"stage": "completion", "completion_sizes": [{"builder": "compact", "size": 2}]}

    database.set_generation_plan(1, plan)

    assert database.get_generation_plan(1) == plan


def test_generation_uses_reproduction_then_mutation_then_completion(tmp_path, monkeypatch):
    population = PopulationManager(
        {
            "initial": {
                "total_size": 1,
                "builder_allocations": [{"builder": "first", "size": 1}],
            },
            "generation": {
                "total_size": 4,
                "reproduction": {"size": 2, "maximum_attempts": 1},
                "mutation": {"size": 1, "maximum_attempts": 1},
                "completion": {
                    "builder_proportions": [
                        {"builder": "first", "proportion": 0.5},
                        {"builder": "second", "proportion": 0.5},
                    ]
                },
            },
        }
    )
    population.population = SimpleNamespace(get_one_candidate=lambda **kwargs: Atoms("H"))
    monkeypatch.setattr(population, "_reproduce", lambda *args, **kwargs: None)

    class Mutation:
        def get_new_individual(self, parents):
            return Atoms("Li"), "mutation: direct"

    database = GlobalOptimisationDatabase(tmp_path / "stages.db")
    database.init_task(
        Atoms("H"),
        data={"population_size": 4, "initial_population_size": 1, "num_atoms_substrate": 1},
    )
    builders = {"first": FixedBuilder("He"), "second": FixedBuilder("Ne")}
    operators = {"mobile": {"mutations": Mutation()}}

    candidates = population._prepare_current_population(database, 1, builders, operators)

    assert [atoms.get_chemical_formula() for atoms in candidates] == ["Li", "He", "He", "Ne"]
    assert database.get_generation_plan(1)["stage"] == "complete"
    reloaded = population._get_current_candidates(database, 1)
    assert len(reloaded["mutated"]) == 1
    assert len(reloaded["completion"]) == 3
    assert len(population._prepare_current_population(database, 1, builders, operators, reloaded)) == 4


def test_monte_carlo_serialization_uses_recipe_and_runtime():
    engine = object.__new__(MonteCarlo)
    engine.random_seed = 11
    engine.builder = Serializable({"method": "builder"})
    engine.worker = Serializable({"schema_version": 2})
    engine.worker.runtime = SimpleNamespace(provider_potential=object())
    engine.operators = [Serializable({"method": "move"})]
    engine.convergence = {"steps": 5}
    engine.dump_period = 2
    engine.ckpt_period = 10
    engine.ignore_atoms_tags = True
    engine.should_retry = False
    engine.restart = False

    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 11
    assert config["recipe"]["operators"] == [{"method": "move"}]


def test_other_global_optimisers_serialize_the_recipe():
    worker = Serializable({"schema_version": 2})

    concurrent = object.__new__(ConcurrentHopping)
    concurrent.random_seed = 13
    concurrent.worker = worker
    concurrent._init_params = {
        "builder": Serializable({"method": "builder"}),
        "convergence": {"generation": 2},
        "use_archive": True,
    }
    concurrent_config = concurrent.as_dict()

    annealing = object.__new__(SimulatedAnnealing)
    annealing.random_seed = 19
    annealing.worker = worker
    annealing.builder = Serializable({"method": "builder"})
    annealing.temperatures = [800.0, 400.0]
    annealing_config = annealing.as_dict()

    assert concurrent_config["recipe"]["random_seed"] == 13
    assert concurrent_config["recipe"]["builder"] == {"method": "builder"}
    assert annealing_config == {
        "method": "simulated_annealing",
        "recipe": {
            "random_seed": 19,
            "builder": {"method": "builder"},
            "temperatures": [800.0, 400.0],
        },
        "runtime": {"schema_version": 2},
    }
