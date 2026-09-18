from types import SimpleNamespace

import pytest

from gdpx.exploration import REGISTER
from gdpx.exploration.factory import create_expedition
from gdpx.exploration.genetic_algorithm.engine import (
    GeneticAlgorithmBroadcaster,
    GeneticAlgorithmEngine,
)
from gdpx.exploration.genetic_algorithm.population.manager import PopulationManager
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
            "random_generator": {"method": "unused"},
            "initial": {"size": 1},
            "generation": {"size": 1},
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
            "initial": {"size": 4},
            "generation": {
                "size": 4,
                "reproduction": 2,
                "random": 1,
                "mutation": 1,
                "maximum_random_attempts": 12,
                "maximum_reproduction_attempts": 24,
            },
            "reproduction": {
                "mutation_probability": 0.25,
                "custom_mutation_probability": 0.75,
            },
            "substrate": {"distance_tolerance": 0.1},
        }
    )

    assert population.init_size == 4
    assert population.gen_rep_size == 2
    assert population.gen_ran_size == 1
    assert population.gen_mut_size == 1
    assert population.gen_ran_max_try == 12
    assert population.gen_rep_max_try == 24
    assert population.pmut == 0.25
    assert population.pmut_custom == 0.75
    assert population.substrate_dtol == 0.1


@pytest.mark.parametrize(
    "population",
    [
        {"init": {"size": 4}},
        {"generation": {"size": 4, "reprod": 4}},
        {"substrate": {"dtol": 0.1}},
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
    engine.generator = Serializable({"method": "random_generator"})
    engine.worker = Serializable({"schema_version": 2})
    engine.ga_dict = {
        "database": "search.db",
        "population": {
            "initial": {"size": 1},
            "generation": {"size": 1},
        },
        "operators": {},
        "property": {"target": "energy"},
        "convergence": {"generation": 1},
        "use_archive": True,
    }

    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 7
    assert config["recipe"]["population"]["random_generator"] == {"method": "random_generator"}
    assert "params" not in config
    assert "worker" not in config


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
