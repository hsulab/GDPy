from types import SimpleNamespace

import pytest

from gdpx.exploration import REGISTER
from gdpx.exploration.factory import create_expedition
from gdpx.exploration.genetic_algorithm.engine import (
    GeneticAlgorithmBroadcaster,
    GeneticAlgorithmEngine,
)
from gdpx.exploration.monte_carlo.concurrent_hopping import ConcurrentHopping
from gdpx.exploration.monte_carlo.monte_carlo import MonteCarlo
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
        builder={"method": "unused"},
        population={"init": {"size": 1}, "gen": {"size": 1}},
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


def test_ga_serialization_uses_recipe_and_runtime():
    engine = object.__new__(GeneticAlgorithmEngine)
    engine.random_seed = 7
    engine.generator = Serializable({"method": "builder"})
    engine.worker = Serializable({"schema_version": 2})
    engine.ga_dict = {
        "database": "search.db",
        "population": {},
        "operators": {},
        "property": {"target": "energy"},
        "convergence": {"generation": 1},
        "use_archive": True,
    }

    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 7
    assert config["recipe"]["builder"] == {"method": "builder"}
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
